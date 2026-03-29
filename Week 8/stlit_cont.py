"""
ASL Sign Recognition — Streamlit App (continuous streaming + gesture spotting)

Usage:
    streamlit run stlit_cont.py
"""

import time
import threading
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
import av
import streamlit as st
from pathlib import Path
from collections import deque
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from asl_citizen_processor import Extractor, FEATURE_DIM
from how2sign.lstm_model import Video_LSTM_morelayers


st.set_page_config(page_title="ASL Sign Recognition", page_icon="🤟", layout="wide")


# config
PROCESSED_DIR = "asl_citizen_processed"
MODEL_PATH    = "saved_models/asl_citizen_fc_model.pth"
HIDDEN_SIZE   = 256
N_LAYERS      = 4
DROPOUT       = 0.5
TOP_K         = 3

# states
IDLE     = "idle"
SIGNING  = "signing"
COOLDOWN = "cooldown"

# spotting thresholds
WINDOW_SIZE     = 45
STRIDE          = 5
CONF_THRESHOLD  = 0.65
COOLDOWN_SEC    = 1.2
MIN_HAND_FRAMES = 10

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# load model
# st.cache_resource means function only runs once
# model stays loaded in memory and gets reused
@st.cache_resource
def load_resources():
    processed_path = Path(PROCESSED_DIR)
    label_map      = pd.read_csv(processed_path / "label_map.csv")
    label_to_gloss = dict(zip(label_map["label"].astype(int), label_map["gloss"]))
    num_classes    = len(label_to_gloss)

    config_path = processed_path / "config.csv"
    feature_dim = (int(pd.read_csv(config_path).iloc[0]["feature_dim"])
                   if config_path.exists() else FEATURE_DIM)

    device = (torch.device("mps")  if torch.backends.mps.is_available()  else
              torch.device("cuda") if torch.cuda.is_available()           else
              torch.device("cpu"))

    model = Video_LSTM_morelayers(
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        num_layers=N_LAYERS,
        num_classes=num_classes,
        input_size=feature_dim,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    model = model.to(device)

    return model, label_to_gloss, device


def draw_landmarks(frame, result):
    h, w = frame.shape[:2]
    for hand_lms in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 220, 120), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)


def maybe_predict(frame_buffer, model, label_to_gloss,
                  top_k, device, results, history, force=False):
    valid = [f for f in frame_buffer if f is not None]
    if len(valid) < 5:
        return False

    video = torch.stack(valid).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _, _ = model(video)
        probs = F.softmax(logits, dim=1)[0]

    k = min(top_k, len(label_to_gloss))
    top_probs, top_idx = probs.topk(k)
    top_results = [(label_to_gloss[i.item()], p.item())
                   for i, p in zip(top_idx, top_probs)]

    conf = top_results[0][1]

    if force or conf >= CONF_THRESHOLD:
        results.clear()
        results.extend(top_results)
        history.append(top_results[0][0])
        print(f"→ {top_results[0][0]}  ({conf*100:.1f}%)")
        return True

    return False

# load model at startup
try:
    model, label_to_gloss, device = load_resources()
except Exception as e:
    st.error(f"Could not load model/data: {e}")
    st.stop()


# shared state
# TWO threads running: 
# Streamlit UI thread (renders page, button clicks) 
# and webrtc recv() thread (processes each frame)
#need this shared dict, (lock). prevents both threads from reading
# and writing at the same time
@st.cache_resource
def get_shared_state():
    return {
        "lock":           threading.Lock(),
        "app_state":      IDLE,
        "frame_buffer":   deque(maxlen=WINDOW_SIZE),
        "frame_counter":  0,
        "cooldown_start": 0.0,
        "results":        [], # latest top-k predictions
        "history":        deque(maxlen=8),
    }

shared = get_shared_state()


# video processor
# calls recv() on every webcam frame
class ASLProcessor:
    def __init__(self):
        # start the mediapipe extractor
        self.extractor = Extractor().__enter__()

    def recv(self, frame):
        # frame comes in as webrtc frame object, convert to numpy arr for cv2
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        now = time.time()

        # tensor = feature vector (None if no hands)
        tensor, result = self.extractor.extract_with_result(img)
        if result is not None:
            draw_landmarks(img, result)

        # touches shared state, need to lock
        with shared["lock"]:
            shared["frame_buffer"].append(tensor)
            shared["frame_counter"] += 1
            frame_counter = shared["frame_counter"]
            hand_frames   = sum(1 for f in shared["frame_buffer"] if f is not None)
            hands_now     = result is not None
            state         = shared["app_state"]

            # state machine — same as webcamcont.py
            if state == COOLDOWN:
                if now - shared["cooldown_start"] > COOLDOWN_SEC:
                    shared["app_state"] = IDLE

            elif state == IDLE:
                if hands_now:
                    shared["app_state"] = SIGNING

            elif state == SIGNING:
                if not hands_now:
                    # hands disappeared -> sign ended -> force predict
                    maybe_predict(
                        shared["frame_buffer"], model, label_to_gloss,
                        TOP_K, device, shared["results"], shared["history"],
                        force=True
                    )
                    shared["app_state"]      = COOLDOWN
                    shared["cooldown_start"] = now

                elif frame_counter % STRIDE == 0 and hand_frames >= MIN_HAND_FRAMES:
                    # mid sign check
                    emitted = maybe_predict(
                        shared["frame_buffer"], model, label_to_gloss,
                        TOP_K, device, shared["results"], shared["history"],
                        force=False
                    )
                    if emitted:
                        shared["app_state"]      = COOLDOWN
                        shared["cooldown_start"] = now

            # overlay state on frame
            # can't update stlit UI elements from inside recv() since
            # on a diff thread. Use cv2 instead
            h, w  = img.shape[:2]
            state = shared["app_state"]
            if state == SIGNING:
                color = (0, 160, 60)
                msg   = f"● SIGNING  ({hand_frames} frames)"
            elif state == COOLDOWN:
                color = (0, 100, 200)
                msg   = "✓ Predicted"
            else:
                color = (30, 30, 30)
                msg   = "Show a sign to begin"

            cv2.rectangle(img, (0, 0), (w, 44), color, -1)
            cv2.putText(img, msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # hand fill bar
            frac = min(1.0, hand_frames / WINDOW_SIZE)
            cv2.rectangle(img, (0, 44), (int(frac * w), 50), (0, 200, 80), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# layout (UI)
st.title("ASL Sign Recognition")

col_cam, col_panel = st.columns([3, 2], gap="large")

with col_cam:
    # webrtc_streamer renders the webcam feed and wires up ASLProcessor.recv()
    # key="asl" uniquely identifies this streamer so Streamlit can track it
    webrtc_streamer(
        key="asl",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=ASLProcessor,
        async_processing=True,
    )

    if st.button("↺  Clear History", use_container_width=True):
        with shared["lock"]:
            shared["results"]       = []
            shared["history"]       = deque(maxlen=8)
            shared["app_state"]     = IDLE
            shared["frame_buffer"]  = deque(maxlen=WINDOW_SIZE)

with col_panel:
    # st.empty() reserves a spot in the UI that can be overwritten later.
    # without this, each rerun would just append new text below the old text
    # instead of replacing it.
    st.subheader("Predictions")
    prediction_placeholder = st.empty()
    st.subheader("History")
    history_placeholder = st.empty()


# render results
# only reruns when user intereacts, but need panel to update automatically
# while true loop keeps script runnign, polls shared state every .5 secs
while True:
    with shared["lock"]:
        results = list(shared["results"])
        history = list(shared["history"])
        state   = shared["app_state"]

    st.caption(f"state: {state}")

    if not results:
        prediction_placeholder.markdown("*No prediction yet — just show a sign!*")
    else:
        lines = []
        for i, (gloss, prob) in enumerate(results):
            prefix = "→" if i == 0 else "  "
            lines.append(f"{prefix} **{gloss}** — {prob*100:.1f}%")
        prediction_placeholder.markdown("\n\n".join(lines))

    if not history:
        history_placeholder.markdown("*Your recent signs will appear here.*")
    else:
        history_placeholder.markdown("  ·  ".join(reversed(history)))

    # can change
    time.sleep(0.5)