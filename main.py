
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import ctypes
import sys
from pathlib import Path
from pycaw.pycaw import AudioUtilities

# =====================================================
# AI GESTURE STUDIO - STEP 5
# Windows 11 | OpenCV | MediaPipe | Pycaw
# =====================================================

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "models" / "hand_landmarker.task"
)

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Volume calibration (640px reference)
MIN_DISTANCE = 15
MAX_DISTANCE = 197

# Smoothing: higher = smoother but slower
SMOOTHING_ALPHA = 0.85

# Ignore very small volume differences
VOLUME_DEADBAND = 1.5
VOLUME_INTERVAL = 0.10

# Discrete gesture safety
GESTURE_HOLD_TIME = 1.0
GESTURE_COOLDOWN = 1.2
NEUTRAL_REARM_TIME = 0.35

# Finger angle thresholds in degrees
EXTENDED_ANGLE = 155
FOLDED_ANGLE = 120

# Colors (OpenCV uses BGR)
BG = (28, 19, 12)
PANEL = (49, 35, 22)
GREEN = (187, 224, 69)
WHITE = (245, 245, 245)
GRAY = (185, 166, 146)
PINK = (225, 64, 224)
BLUE = (255, 150, 55)
YELLOW = (70, 210, 255)

# Finger joint indices: MCP, PIP, TIP
FINGERS = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# =====================================================
# UI HELPERS
# =====================================================

def text(frame, message, x, y,
         scale=0.65, color=WHITE, thickness=2):

    cv2.putText(
        frame,
        str(message),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def panel(frame, x1, y1, x2, y2, color=PANEL):

    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        cv2.FILLED
    )


def draw_dashboard(frame, volume, muted,
                   fps, gesture, mode,
                   progress, message):

    h, w = frame.shape[:2]

    # Dark transparent header
    overlay = frame.copy()
    panel(overlay, 0, 0, w, 100, BG)

    cv2.addWeighted(
        overlay, 0.84,
        frame, 0.16,
        0, frame
    )

    text(
        frame, "GESTURE STUDIO",
        25, 42, 0.95, GREEN
    )

    text(
        frame, "AI HAND CONTROL / STEP 05",
        27, 74, 0.48, GRAY, 1
    )

    text(
        frame,
        f"FPS {int(fps)}",
        w - 135, 48, 0.6, WHITE
    )

    # Side dashboard
    side_width = min(295, int(w * 0.32))
    side_x = w - side_width - 15

    overlay = frame.copy()

    panel(
        overlay,
        side_x, 115,
        w - 15, h - 70
    )

    cv2.addWeighted(
        overlay, 0.9,
        frame, 0.1,
        0, frame
    )

    x = side_x + 22

    text(
        frame, "SYSTEM AUDIO",
        x, 155, 0.6, GRAY
    )

    display_volume = int(round(volume))

    if muted:
        display_text = "MUTED"
    else:
        display_text = f"{display_volume}%"

    text(
        frame, display_text,
        x, 215, 1.3, GREEN, 3
    )

    text(
        frame, f"MODE: {mode}",
        x, 265, 0.52, WHITE
    )

    text(
        frame, f"GESTURE: {gesture}",
        x, 310, 0.48, WHITE
    )

    # Volume bar
    bar_top = 350
    bar_bottom = h - 185

    if bar_bottom > bar_top:

        bar_x1 = x
        bar_x2 = x + 40

        cv2.rectangle(
            frame,
            (bar_x1, bar_top),
            (bar_x2, bar_bottom),
            GRAY, 2
        )

        level = 0 if muted else volume

        fill_y = int(
            np.interp(
                level,
                [0, 100],
                [bar_bottom, bar_top]
            )
        )

        if fill_y < bar_bottom - 2:

            cv2.rectangle(
                frame,
                (bar_x1 + 3, fill_y),
                (bar_x2 - 3, bar_bottom - 3),
                GREEN, cv2.FILLED
            )

        text(
            frame, "0 - 100",
            x + 55, bar_bottom,
            0.48, GRAY, 1
        )

    # Gesture hold progress
    progress = float(np.clip(progress, 0, 1))

    progress_y = h - 145
    progress_width = side_width - 44

    text(
        frame,
        "GESTURE CONFIRMATION",
        x, progress_y - 16,
        0.44, GRAY, 1
    )

    cv2.rectangle(
        frame,
        (x, progress_y),
        (x + progress_width, progress_y + 12),
        (87, 73, 60),
        cv2.FILLED
    )

    cv2.rectangle(
        frame,
        (x, progress_y),
        (
            x + int(progress_width * progress),
            progress_y + 12
        ),
        GREEN,
        cv2.FILLED
    )

    # Status message
    text(
        frame, message[:32],
        x, h - 95,
        0.48, YELLOW, 1
    )

    # Footer
    overlay = frame.copy()
    panel(overlay, 0, h - 52, w, h, BG)

    cv2.addWeighted(
        overlay, 0.90,
        frame, 0.10,
        0, frame
    )

    text(
        frame,
        "C: ENABLE  |  P: PAUSE  |  Q: EXIT",
        25, h - 19,
        0.57, WHITE, 1
    )


# =====================================================
# HAND LANDMARKS
# =====================================================

def draw_hand(frame, landmarks):

    h, w = frame.shape[:2]
    points = []

    for landmark in landmarks:

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        points.append((x, y))

    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            WHITE, 2
        )

    for i, point in enumerate(points):

        color = GREEN
        radius = 5

        if i == 4:
            color = PINK
            radius = 10

        elif i == 8:
            color = BLUE
            radius = 10

        cv2.circle(
            frame,
            point,
            radius,
            color,
            cv2.FILLED
        )

    return points


# =====================================================
# ANGLE-BASED FINGER RECOGNITION
# =====================================================

def joint_angle(a, b, c):

    # Angle ABC, with B being the middle joint
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return 0.0

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return math.degrees(
        math.acos(cosine)
    )


def recognize_gesture(points):

    states = {}

    for name, (mcp, pip, tip) in FINGERS.items():

        angle = joint_angle(
            points[mcp],
            points[pip],
            points[tip]
        )

        if angle >= EXTENDED_ANGLE:
            states[name] = "extended"

        elif angle <= FOLDED_ANGLE:
            states[name] = "folded"

        else:
            states[name] = "partial"

    # Open palm: four straight fingers
    if all(
        state == "extended"
        for state in states.values()
    ):
        return "PLAY_PAUSE"

    # Fist: four folded fingers
    if all(
        state == "folded"
        for state in states.values()
    ):
        return "MUTE"

    # Volume: only index extended
    if (
        states["index"] == "extended"
        and states["middle"] == "folded"
        and states["ring"] == "folded"
        and states["pinky"] == "folded"
    ):
        return "VOLUME"

    return "NONE"


# =====================================================
# WINDOWS MEDIA PLAY / PAUSE
# =====================================================

def toggle_media_play_pause():

    # Windows VK_MEDIA_PLAY_PAUSE
    VK_MEDIA_PLAY_PAUSE = 0xB3
    KEYEVENTF_KEYUP = 0x0002

    user32 = ctypes.windll.user32

    user32.keybd_event(
        VK_MEDIA_PLAY_PAUSE,
        0, 0, 0
    )

    user32.keybd_event(
        VK_MEDIA_PLAY_PAUSE,
        0, KEYEVENTF_KEYUP, 0
    )


# =====================================================
# MAIN APPLICATION
# =====================================================

def main():

    if not MODEL_PATH.is_file():
        print("ERROR: Model file not found")
        print(MODEL_PATH)
        return

    # Windows default speaker
    try:

        device = AudioUtilities.GetSpeakers()
        audio = device.EndpointVolume

        print("Audio Device:", device.FriendlyName)

    except Exception as error:

        print("Audio initialization error:", error)
        return

    # Webcam
    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    # MediaPipe Tasks API
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    initial_volume = (
        audio.GetMasterVolumeLevelScalar() * 100
    )

    smooth_volume = initial_volume
    was_volume_gesture = False

    control_enabled = False

    last_volume_update = 0.0

    # Discrete gesture state
    candidate = "NONE"
    candidate_since = 0.0

    gesture_armed = True
    neutral_since = None
    last_action_time = -100.0

    message = "Press C to enable"

    previous_time = time.perf_counter()
    start_time = time.perf_counter()
    last_timestamp_ms = -1

    print("Gesture Studio Started!")
    print("C = Enable")
    print("P = Pause")
    print("Q = Exit")

    try:

        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:

                ok, frame = cap.read()

                if not ok:
                    print("Camera frame error")
                    break

                frame = cv2.flip(frame, 1)

                h, w = frame.shape[:2]

                now = time.perf_counter()

                # Smaller image for AI processing
                processing_width = 640

                processing_height = int(
                    h * processing_width / w
                )

                small_frame = cv2.resize(
                    frame,
                    (processing_width, processing_height)
                )

                rgb = cv2.cvtColor(
                    small_frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms = int(
                    (now - start_time) * 1000
                )

                timestamp_ms = max(
                    timestamp_ms,
                    last_timestamp_ms + 1
                )

                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                gesture = "NONE"
                progress = 0.0

                if result.hand_landmarks:

                    landmarks = result.hand_landmarks[0]

                    points = draw_hand(
                        frame,
                        landmarks
                    )

                    gesture = recognize_gesture(points)

                    thumb = points[4]
                    index = points[8]

                    # Connect thumb and index
                    cv2.line(
                        frame,
                        thumb, index,
                        PINK, 3
                    )

                    # Finger distance
                    distance = math.hypot(
                        thumb[0] - index[0],
                        thumb[1] - index[1]
                    )

                    normalized_distance = (
                        distance * 640.0 / w
                    )

                    text(
                        frame,
                        f"Finger gap: {int(normalized_distance)}",
                        25, 145,
                        0.65, WHITE
                    )

                    # ---------------- VOLUME ----------------

                    if (
                        control_enabled
                        and gesture == "VOLUME"
                    ):

                        target_volume = float(
                            np.interp(
                                normalized_distance,
                                [MIN_DISTANCE, MAX_DISTANCE],
                                [0, 100]
                            )
                        )

                        if not was_volume_gesture:

                            smooth_volume = (
                                audio.GetMasterVolumeLevelScalar()
                                * 100
                            )

                        smooth_volume = (
                            SMOOTHING_ALPHA * smooth_volume
                            + (1 - SMOOTHING_ALPHA) * target_volume
                        )

                        actual_volume = (
                            audio.GetMasterVolumeLevelScalar()
                            * 100
                        )

                        difference = abs(
                            smooth_volume - actual_volume
                        )

                        if (
                            difference >= VOLUME_DEADBAND
                            and now - last_volume_update
                            >= VOLUME_INTERVAL
                        ):

                            audio.SetMasterVolumeLevelScalar(
                                float(
                                    np.clip(
                                        smooth_volume / 100,
                                        0, 1
                                    )
                                ),
                                None
                            )

                            last_volume_update = now

                        message = "Adjusting volume"

                    was_volume_gesture = (
                        control_enabled
                        and gesture == "VOLUME"
                    )

                else:
                    was_volume_gesture = False

                # ---------------- DISCRETE GESTURES ----------------

                discrete = (
                    gesture
                    if control_enabled and gesture in (
                        "MUTE", "PLAY_PAUSE"
                    )
                    else "NONE"
                )

                if discrete == "NONE":

                    candidate = "NONE"

                    if neutral_since is None:
                        neutral_since = now

                    if (
                        now - neutral_since
                        >= NEUTRAL_REARM_TIME
                    ):
                        gesture_armed = True

                else:

                    neutral_since = None

                    if discrete != candidate:

                        candidate = discrete
                        candidate_since = now

                    held = now - candidate_since

                    if gesture_armed:
                        progress = min(
                            held / GESTURE_HOLD_TIME,
                            1.0
                        )

                    if (
                        gesture_armed
                        and held >= GESTURE_HOLD_TIME
                        and now - last_action_time
                        >= GESTURE_COOLDOWN
                    ):

                        if discrete == "MUTE":

                            current_mute = audio.GetMute()

                            audio.SetMute(
                                0 if current_mute else 1,
                                None
                            )

                            message = (
                                "Unmuted"
                                if current_mute
                                else "Muted"
                            )

                        elif discrete == "PLAY_PAUSE":

                            toggle_media_play_pause()

                            message = (
                                "Play/Pause command sent"
                            )

                        last_action_time = now
                        gesture_armed = False
                        progress = 1.0

                # ---------------- STATUS ----------------

                actual_volume = (
                    audio.GetMasterVolumeLevelScalar()
                    * 100
                )

                muted = bool(audio.GetMute())

                elapsed = (
                    time.perf_counter() - previous_time
                )

                fps = (
                    1.0 / elapsed
                    if elapsed > 0 else 0
                )

                previous_time = time.perf_counter()

                mode = (
                    "ACTIVE"
                    if control_enabled
                    else "PAUSED"
                )

                draw_dashboard(
                    frame,
                    actual_volume,
                    muted,
                    fps,
                    gesture,
                    mode,
                    progress,
                    message
                )

                # Show video
                cv2.imshow(
                    "Gesture Studio - Premium Edition",
                    frame
                )

                # ---------------- KEYBOARD ----------------

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                elif key == ord("c"):

                    control_enabled = True
                    was_volume_gesture = False

                    candidate = "NONE"
                    gesture_armed = True
                    neutral_since = None

                    message = "Control enabled"

                elif key == ord("p"):

                    control_enabled = False
                    was_volume_gesture = False

                    candidate = "NONE"
                    gesture_armed = False
                    neutral_since = None

                    message = "Control paused"

    finally:

        cap.release()
        cv2.destroyAllWindows()

        print("Gesture Studio closed.")


if __name__ == "__main__":
    main()
