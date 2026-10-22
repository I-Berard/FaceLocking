# Gap Analysis: Whole-Person Tracking & Face-Association Module

## Assignment Requirements vs. Current Implementation

| Requirement | Status | Details |
|---|---|---|
| **Webcam-based face detection** | ✅ Implemented | Haar + MediaPipe FaceLandmarker pipeline |
| **Face recognition / identity** | ✅ Implemented | ArcFace ONNX embedding + cosine DB matching |
| **Face enrollment** | ✅ Implemented | [`enroll.py`](file:///home/berard/Robotics/face-recognition-5pt/src/enroll.py) with auto/manual capture |
| **Target-locked face tracking** | ✅ Implemented | [`face_tracking.py`](file:///home/berard/Robotics/face-recognition-5pt/src/face_tracking.py) — `LockedFaceTracker` |
| **Facial signal analysis (EAR, blinks, smiles)** | ✅ Implemented | [`face_signals.py`](file:///home/berard/Robotics/face-recognition-5pt/src/face_signals.py) |
| **Target centring logic** | ✅ Implemented | `position_signal()` with dead-zone + EMA smoothing |
| **CPU-only Python** | ✅ Implemented | ONNX Runtime CPU, OpenCV, MediaPipe |
| **Three-state model (Confirmed / Temporary / Unknown)** | ❌ **Missing** | Has `SEARCHING / LOCKED / LOST` — no "Temporary" or "Unknown" with confidence scoring |
| **Exponential confidence decay** | ❌ **Missing** | Uses a simple frame counter (`lost_frames`) with hard timeout — no decay function |
| **Whole-person (body) detection** | ❌ **Missing** | Only detects faces, no full-body bounding boxes |
| **Spatial face-to-body mapping** | ❌ **Missing** | No body detector exists to associate faces with |
| **Temporal identity continuity across facial loss** | ⚠️ Partial | The tracker uses IoU/displacement to follow a box, but identity is lost after `lost_timeout` frames with no confidence decay bridge |
| **Head turn / occlusion handling** | ⚠️ Partial | LOST state keeps `last_box` briefly, but no body-level fallback when face vanishes |
| **Path crossing handling** | ❌ **Missing** | No multi-person body tracking — face IoU alone can't handle people crossing paths |
| **Unit tests for real-world scenarios** | ⚠️ Partial | [`test_face_tracking.py`](file:///home/berard/Robotics/face-recognition-5pt/tests/test_face_tracking.py) tests exclusive focus, but no tests for the three-state model, decay, or body tracking |

---

## What's Missing — Summary

### 1. Three-State Identity Model
The assignment explicitly requires **Confirmed**, **Temporary**, and **Unknown** states:
- **Confirmed**: Face recognized with high confidence (currently closest to `LOCKED`)
- **Temporary**: Face lost but body still tracked — identity maintained with decaying confidence
- **Unknown**: No face or body match — identity dropped

Current [`LockState`](file:///home/berard/Robotics/face-recognition-5pt/src/face_tracking.py#L29-L32) has `SEARCHING / LOCKED / LOST`, which is conceptually close but lacks the confidence-decay semantics.

### 2. Exponential Confidence Decay
When a face is lost (head turn, occlusion), confidence should decay exponentially:
```
confidence(t) = initial_confidence × e^(-λt)
```
Currently the code uses a simple integer counter (`lost_frames > lost_timeout`), with no gradual confidence value.

### 3. Whole-Person (Body) Detection
No body detector exists. The assignment requires person-level bounding boxes so that when a face disappears (head turn), the system can continue tracking the **body** and re-associate the face when it reappears.

### 4. Spatial Face-to-Body Mapping
With no body detector, there's no logic to map "this face belongs inside this body bounding box." This is critical for multi-person scenarios and path crossings.

---

## Implementation Plan

### Phase 1 — Body Detection via HOG Person Detector
Use OpenCV's built-in `cv2.HOGDescriptor` with `getDefaultPeopleDetector()` — pure CPU, no extra models needed.

### Phase 2 — Three-State Tracker with Exponential Decay
Replace `LockState` with a proper `IdentityState(Confirmed, Temporary, Unknown)` model:
- Track `confidence: float` per person
- On face match: state → Confirmed, confidence → 1.0
- On face loss but body present: state → Temporary, confidence decays as `c × e^(-λΔt)`
- On confidence below threshold or body lost: state → Unknown

### Phase 3 — Face-to-Body Spatial Association
- For each detected face, find the body bounding box that contains it (IoU or containment check)
- When multiple people are present, this mapping prevents identity swaps during path crossings

### Phase 4 — Multi-Person Tracker
- Maintain a dict of tracked persons, each with their own state machine
- Hungarian algorithm or greedy matching for frame-to-frame body association

### Phase 5 — Comprehensive Tests
- Head turn scenario (face lost → body tracked → face returns)
- Occlusion scenario (person walks behind obstacle)
- Path crossing scenario (two people swap positions)
- Confidence decay verification (exponential curve)

> [!IMPORTANT]
> The project has strong foundations (face detection, recognition, alignment, embedding, target tracking), but is missing the **body-level tracking layer** and the **three-state confidence model** that the assignment specifically requires.

Shall I proceed with the implementation?
