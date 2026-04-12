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
import requests
import streamlit as st
from pathlib import Path
from collections import deque
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from asl_citizen_processor import Extractor, FEATURE_DIM
from how2sign.lstm_model import Video_LSTM_morelayers


st.set_page_config(page_title="ASL Sign Recognition", page_icon="🤟", layout="wide")


# ── config ────────────────────────────────────────────────────────────────────

WORD_PROCESSED_DIR = "asl_citizen_processed"
WORD_MODEL_PATH    = "saved_models/asl_citizen_fc_model.pth"

FS_PROCESSED_DIR = "fingerspelling_all_processed"
FS_MODEL_PATH    = "saved_models/fingerspelling_fc_model.pth"

HIDDEN_SIZE = 256
N_LAYERS    = 4
DROPOUT     = 0.5
TOP_K       = 3

# Translation server — update when Colab restarts
TRANSLATE_URL = "https://mutt-headless-bacon.ngrok-free.dev"

TRANSLATE_SYSTEM_PROMPT = """
You are an ASL gloss to English translator.
Translate the ASL gloss the user gives you into a single natural English sentence.
Output ONLY the translated sentence. Nothing else. No explanations, no examples, no extra text.
"""

# How long to wait after the last sign before auto-translating (seconds)
TRANSLATE_DEBOUNCE_SEC = 2.0

# modes
MODE_WORD   = "Word Signs"
MODE_FINGER = "Fingerspelling"

# states
IDLE     = "idle"
SIGNING  = "signing"
COOLDOWN = "cooldown"

# spotting thresholds
WORD_WINDOW_SIZE     = 45
WORD_MIN_HAND_FRAMES = 10
FS_WINDOW_SIZE       = 20
FS_MIN_HAND_FRAMES   = 4

STRIDE         = 5
CONF_THRESHOLD = 0.65
COOLDOWN_SEC   = 1.2

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ── load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_word_resources():
    processed_path = Path(WORD_PROCESSED_DIR)
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
    model.load_state_dict(torch.load(WORD_MODEL_PATH, map_location="cpu"))
    model.eval()
    model = model.to(device)
    return model, label_to_gloss, device


@st.cache_resource
def load_fs_resources():
    processed_path = Path(FS_PROCESSED_DIR)
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
    model.load_state_dict(torch.load(FS_MODEL_PATH, map_location="cpu"))
    model.eval()
    model = model.to(device)
    return model, label_to_gloss, device


# load both at startup
try:
    word_model, word_label_to_gloss, device = load_word_resources()
except Exception as e:
    st.error(f"Could not load word sign model: {e}")
    st.stop()

try:
    fs_model, fs_label_to_gloss, _ = load_fs_resources()
    fs_available = True
except Exception:
    fs_available = False


# ── shared state ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_shared_state():
    return {
        "lock":              threading.Lock(),
        "mode":              MODE_WORD,
        "app_state":         IDLE,
        "frame_buffer":      deque(maxlen=WORD_WINDOW_SIZE),
        "frame_counter":     0,
        "cooldown_start":    0.0,
        "results":           [],
        "history":           deque(maxlen=20),
        "last_sign_time":    0.0,
        "pending_translate": False,  # True when a new sign was committed
    }

shared = get_shared_state()


# ── helpers ───────────────────────────────────────────────────────────────────
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


def build_sentence(history: list, mode: str) -> str:
    """
    Consecutive letters merge into words (H,E,L,L,O → HELLO).
    Word signs have trailing digits stripped (DOG1 → DOG).
    """
    if not history:
        return ""

    tokens = []
    for sign in history:
        if len(sign) == 1 and sign.isalpha():
            tokens.append(("letter", sign))
        else:
            clean = sign.rstrip("0123456789")
            tokens.append(("word", clean))

    result = ""
    i = 0
    while i < len(tokens):
        kind, val = tokens[i]
        if kind == "letter":
            chunk = val
            while i + 1 < len(tokens) and tokens[i + 1][0] == "letter":
                i += 1
                chunk += tokens[i][1]
            if result and not result.endswith(" "):
                result += " "
            result += chunk
        else:
            if result and not result.endswith(" "):
                result += " "
            result += val
            result += " "
        i += 1

    return result.strip()


def translate_gloss(gloss_text: str, base_url: str) -> str:
    """
    Sends the gloss sentence to the llama-cpp server and returns
    the English translation.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
            {"role": "user",   "content": gloss_text},
        ],
        "temperature": 0.2,
        "max_tokens":  128,
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── video processor ───────────────────────────────────────────────────────────
class ASLProcessor:
    def __init__(self):
        self.extractor = Extractor().__enter__()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        now = time.time()

        tensor, result = self.extractor.extract_with_result(img)
        if result is not None:
            draw_landmarks(img, result)

        with shared["lock"]:
            mode = shared["mode"]

            if mode == MODE_FINGER and fs_available:
                active_model     = fs_model
                active_label_map = fs_label_to_gloss
                window_size      = FS_WINDOW_SIZE
                min_hand_frames  = FS_MIN_HAND_FRAMES
            else:
                active_model     = word_model
                active_label_map = word_label_to_gloss
                window_size      = WORD_WINDOW_SIZE
                min_hand_frames  = WORD_MIN_HAND_FRAMES

            shared["frame_buffer"].append(tensor)
            shared["frame_counter"] += 1
            frame_counter = shared["frame_counter"]
            hand_frames   = sum(1 for f in shared["frame_buffer"] if f is not None)
            hands_now     = result is not None
            state         = shared["app_state"]

            if state == COOLDOWN:
                if now - shared["cooldown_start"] > COOLDOWN_SEC:
                    shared["app_state"] = IDLE

            elif state == IDLE:
                if hands_now:
                    shared["app_state"] = SIGNING

            elif state == SIGNING:
                if not hands_now:
                    emitted = maybe_predict(
                        shared["frame_buffer"], active_model, active_label_map,
                        TOP_K, device, shared["results"], shared["history"],
                        force=True
                    )
                    if emitted:
                        shared["pending_translate"] = True
                    shared["last_sign_time"] = now
                    shared["app_state"]      = COOLDOWN
                    shared["cooldown_start"] = now

                elif frame_counter % STRIDE == 0 and hand_frames >= min_hand_frames:
                    emitted = maybe_predict(
                        shared["frame_buffer"], active_model, active_label_map,
                        TOP_K, device, shared["results"], shared["history"],
                        force=False
                    )
                    if emitted:
                        shared["pending_translate"] = True
                        shared["last_sign_time"]    = now
                        shared["app_state"]         = COOLDOWN
                        shared["cooldown_start"]    = now

            # overlay
            h, w       = img.shape[:2]
            state      = shared["app_state"]
            mode_color = (180, 120, 0) if mode == MODE_FINGER else (0, 160, 60)

            if state == SIGNING:
                color = mode_color
                msg   = f"● SIGNING  ({hand_frames} frames)  [{mode}]"
            elif state == COOLDOWN:
                color = (0, 100, 200)
                msg   = f"✓ Predicted  [{mode}]"
            else:
                color = (30, 30, 30)
                msg   = f"Show a sign  [{mode}]"

            cv2.rectangle(img, (0, 0), (w, 44), color, -1)
            cv2.putText(img, msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            frac = min(1.0, hand_frames / window_size)
            cv2.rectangle(img, (0, 44), (int(frac * w), 50), (0, 200, 80), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── layout ────────────────────────────────────────────────────────────────────
st.title("ASL Sign Recognition 🤟")

col_cam, col_panel = st.columns([3, 2], gap="large")

with col_cam:
    if fs_available:
        mode_choice = st.radio(
            "Mode",
            [MODE_WORD, MODE_FINGER],
            horizontal=True,
            help="Word Signs: ASL glosses  |  Fingerspelling: A–Z letters"
        )
        with shared["lock"]:
            if shared["mode"] != mode_choice:
                shared["mode"]         = mode_choice
                shared["frame_buffer"] = deque(maxlen=FS_WINDOW_SIZE if mode_choice == MODE_FINGER else WORD_WINDOW_SIZE)
                shared["app_state"]    = IDLE
                shared["results"]      = []
    else:
        st.info("ℹ️ Fingerspelling model not found — word sign mode only.")

    webrtc_streamer(
        key="asl",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=ASLProcessor,
        async_processing=True,
    )

    if st.button("↺  Clear History", use_container_width=True):
        with shared["lock"]:
            shared["results"]           = []
            shared["history"]           = deque(maxlen=20)
            shared["app_state"]         = IDLE
            shared["frame_buffer"]      = deque(maxlen=WORD_WINDOW_SIZE)
            shared["last_sign_time"]    = 0.0
            shared["pending_translate"] = False
        st.session_state["translation"] = ""

    st.markdown("---")
    st.caption("ASL Gloss")
    sentence_placeholder = st.empty()

    st.caption("English Translation")
    translation_placeholder = st.empty()


with col_panel:
    st.subheader("Predictions")
    prediction_placeholder = st.empty()
    st.subheader("History")
    history_placeholder = st.empty()

    st.markdown("---")
    st.subheader("Translation Server")
    server_url = st.text_input(
        "ngrok URL",
        value=TRANSLATE_URL,
        help="Paste the ngrok URL from your Colab notebook here"
    )
    st.caption(f"Auto-translates {TRANSLATE_DEBOUNCE_SEC}s after your last sign.")


# ── session state init ────────────────────────────────────────────────────────
if "translation" not in st.session_state:
    st.session_state["translation"] = ""
if "translating" not in st.session_state:
    st.session_state["translating"] = False


# ── render loop ───────────────────────────────────────────────────────────────
while True:
    with shared["lock"]:
        results   = list(shared["results"])
        history   = list(shared["history"])
        state     = shared["app_state"]
        mode      = shared["mode"]
        pending   = shared["pending_translate"]
        last_time = shared["last_sign_time"]

    st.caption(f"state: {state}  |  mode: {mode}")

    # predictions panel
    if not results:
        prediction_placeholder.markdown("*No prediction yet — just show a sign!*")
    else:
        lines = []
        for i, (gloss, prob) in enumerate(results):
            prefix = "→" if i == 0 else "  "
            lines.append(f"{prefix} **{gloss}** — {prob*100:.1f}%")
        prediction_placeholder.markdown("\n\n".join(lines))

    # history panel
    if not history:
        history_placeholder.markdown("*Your recent signs will appear here.*")
    else:
        history_placeholder.markdown("  ·  ".join(reversed(history)))

    # gloss sentence
    sentence = build_sentence(list(history), mode)
    if sentence:
        sentence_placeholder.markdown(f"### {sentence}")
    else:
        sentence_placeholder.markdown("*Start signing to build a sentence...*")

    # auto-translate: fires when there's a pending sign and the user
    # has paused for TRANSLATE_DEBOUNCE_SEC seconds
    now = time.time()
    if (
        sentence
        and pending
        and (now - last_time) >= TRANSLATE_DEBOUNCE_SEC
        and not st.session_state["translating"]
    ):
        with shared["lock"]:
            shared["pending_translate"] = False

        st.session_state["translating"] = True
        try:
            translation = translate_gloss(sentence, server_url)
            st.session_state["translation"] = translation
        except Exception as e:
            st.session_state["translation"] = f"⚠️ Translation failed: {e}"
        finally:
            st.session_state["translating"] = False

    # show translation
    if st.session_state["translating"]:
        translation_placeholder.markdown("*Translating...*")
    elif st.session_state["translation"]:
        translation_placeholder.markdown(f"### {st.session_state['translation']}")
    else:
        translation_placeholder.markdown("*Translation will appear here automatically.*")

    time.sleep(0.5)