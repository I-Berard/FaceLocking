#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Stepper.h>

// =====================================================
// WIFI CONFIGURATION
// =====================================================

const char* WIFI_SSID = "Main Hall";
const char* WIFI_PASSWORD = "Meeting@2024";

// =====================================================
// MQTT CONFIGURATION
// =====================================================

// IMPORTANT:
// This must be the IP address of the computer running
// your Mosquitto MQTT broker.
const char* MQTT_BROKER = "10.12.74.142";

const int MQTT_PORT = 1883;

// MQTT topic
const char* MQTT_TOPIC = "position";

// =====================================================
// STEPPER MOTOR CONFIGURATION
// =====================================================

// 28BYJ-48 is commonly used with 2048 steps/revolution
const int STEPS_PER_REVOLUTION = 2048;

// ESP8266 -> ULN2003
//
// D1 = GPIO5
// D2 = GPIO4
// D5 = GPIO14
// D6 = GPIO12
//
// The order here is important.
Stepper stepper(
  STEPS_PER_REVOLUTION,
  5,
  14,
  4,
  12
);

// =====================================================
// MOTOR POSITION
// =====================================================

// Motor position currently being used
long currentPosition = 0;

// Maximum movement from the center
//
// -1024 = approximately half revolution left
//    0  = center
// +1024 = approximately half revolution right
//
const long MAX_POSITION = 1024;

// =====================================================
// MQTT CLIENT
// =====================================================

WiFiClient espClient;
PubSubClient mqttClient(espClient);


// =====================================================
// CONNECT TO WIFI
// =====================================================

void connectWiFi()
{
  Serial.println();
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("Wi-Fi connected!");

  Serial.print("ESP8266 IP address: ");
  Serial.println(WiFi.localIP());
}


// =====================================================
// MQTT MESSAGE CALLBACK
// =====================================================

void mqttCallback(char* topic, byte* payload, unsigned int length)
{
  Serial.println();
  Serial.print("MQTT message received on topic: ");
  Serial.println(topic);

  // Convert MQTT payload to a String
  String message = "";

  for (unsigned int i = 0; i < length; i++)
  {
    message += (char)payload[i];
  }

  Serial.print("Received value: ");
  Serial.println(message);

  // Convert String to floating-point number
  float value = message.toFloat();

  Serial.print("Parsed value: ");
  Serial.println(value);

  // Make sure value stays between -1 and +1
  value = constrain(value, -1.0, 1.0);

  // ===================================================
  // CONVERT -1...+1 TO MOTOR POSITION
  // ===================================================

  long targetPosition = -value * MAX_POSITION;

  Serial.print("Target motor position: ");
  Serial.println(targetPosition);

  // ===================================================
  // CALCULATE HOW MANY STEPS TO MOVE
  // ===================================================

  long stepsToMove = targetPosition - currentPosition;

  Serial.print("Steps to move: ");
  Serial.println(stepsToMove);

  // Move the motor
  stepper.step(stepsToMove);

  // Remember the new position
  currentPosition = targetPosition;

  Serial.print("New motor position: ");
  Serial.println(currentPosition);
}


// =====================================================
// CONNECT TO MQTT BROKER
// =====================================================

void reconnectMQTT()
{
  while (!mqttClient.connected())
  {
    Serial.print("Connecting to MQTT broker...");

    // Create a unique MQTT client ID
    String clientID = "ESP8266-Tachometer-";
    clientID += String(ESP.getChipId());

    if (mqttClient.connect(clientID.c_str()))
    {
      Serial.println("connected!");

      // Subscribe to the topic
      if (mqttClient.subscribe(MQTT_TOPIC))
      {
        Serial.print("Subscribed to: ");
        Serial.println(MQTT_TOPIC);
      }
      else
      {
        Serial.println("MQTT subscription failed!");
      }
    }
    else
    {
      Serial.print("Failed, MQTT state = ");
      Serial.println(mqttClient.state());

      Serial.println("Retrying in 5 seconds...");
      delay(5000);
    }
  }
}


// =====================================================
// SETUP
// =====================================================

void setup()
{
  Serial.begin(115200);

  delay(1000);

  Serial.println();
  Serial.println("================================");
  Serial.println("ESP8266 MQTT TACHOMETER");
  Serial.println("================================");

  // Set motor speed
  stepper.setSpeed(10);

  // Connect to Wi-Fi
  connectWiFi();

  // Configure MQTT broker
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);

  // Set MQTT callback
  mqttClient.setCallback(mqttCallback);
}


// =====================================================
// MAIN LOOP
// =====================================================

void loop()
{
  // Check Wi-Fi connection
  if (WiFi.status() != WL_CONNECTED)
  {
    Serial.println("Wi-Fi disconnected!");
    connectWiFi();
  }

  // Check MQTT connection
  if (!mqttClient.connected())
  {
    reconnectMQTT();
  }

  // Process incoming MQTT messages
  mqttClient.loop();
}