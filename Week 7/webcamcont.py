"""
Webcam Inference — ASL Citizen LSTM (continuous streaming + gesture spotting)

Controls:
  C — clear history
  Q — quit
"""

import cv2
import time
import torch
import torch.nn.functional as F
import pandas as pd
import argparse
from pathlib import Path
from collections import deque

from asl_citizen_processor import Extractor, FEATURE_DIM
#from lstm_model import Video_LSTM_morelayers as Video_LSTM
from lstm_model import Video_LSTM


# states
IDLE     = "idle"
SIGNING  = "signing"
COOLDOWN = "cooldown"

# spotting thresholds
WINDOW_SIZE     = 45    # frames kept in rolling buffer (~1.5s at 30fps)
STRIDE          = 5     # run inference every N frames
CONF_THRESHOLD  = 0.65  # min top-1 confidence to emit a prediction
COOLDOWN_SEC    = 1.2   # seconds before another prediction can fire
MIN_HAND_FRAMES = 10    # window needs at least this many hand-detected frames


# model helpers (unchanged)
def load_model(model_path, num_classes, hidden_size, n_layers, dropout, feature_dim, device):
    model = Video_LSTM(
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers=n_layers,
        num_classes=num_classes,
        input_size=feature_dim,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model.to(device)


HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_landmarks(frame, result):
    h, w = frame.shape[:2]
    for hand_lms in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 220, 120), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)
            cv2.circle(frame, pt, 4, (0, 180, 90), 1)


def predict(model, frames, label_to_gloss, top_k, device):
    if len(frames) < 5:
        return [("(too short)", 0.0)]
    video = torch.stack(frames).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _, _ = model(video)
        probs = F.softmax(logits, dim=1)[0]
    k = min(top_k, len(label_to_gloss))
    top_probs, top_idx = probs.topk(k)
    return [(label_to_gloss[i.item()], p.item()) for i, p in zip(top_idx, top_probs)]


# new helpers
def maybe_predict(frame_buffer, model, label_to_gloss,
                   top_k, device, results, history, force=False):
    """
    Pull valid (hand-detected) frames from the buffer and run inference.
    Returns True if a prediction was emitted.
    force=True skips the confidence gate (used when hands just disappeared).
    """
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

    # CONFIDENCE THRESHOLDING: if model isn't sure then returns false and keep collecting frames
    if force or conf >= CONF_THRESHOLD:
        results.clear()
        results.extend(top_results)
        history.append(top_results[0][0])
        print(f"→ {top_results[0][0]}  ({conf*100:.1f}%)")
        return True

    return False


def _draw_continuous_ui(frame, state, results, history, hand_frames):
    h, w = frame.shape[:2]

    # status bar
    if state == SIGNING:
        color = (0, 160, 60)
        msg   = f"● SIGNING  ({hand_frames} frames with hands)"
    elif state == COOLDOWN:
        color = (0, 100, 200)
        msg   = "✓ Predicted — cooldown"
    else:
        color = (30, 30, 30)
        msg   = "Show a sign to begin   C = clear   Q = quit"

    cv2.rectangle(frame, (0, 0), (w, 54), color, -1)
    cv2.putText(frame, msg, (15, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # window fill bar — shows how many hand frames are in the buffer
    frac = min(1.0, hand_frames / WINDOW_SIZE)
    cv2.rectangle(frame, (0, 54), (int(frac * w), 60), (0, 200, 80), -1)

    # predictions panel
    if results:
        panel_top = h - 170
        cv2.rectangle(frame, (0, panel_top), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, "Prediction:", (15, panel_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
        for i, (gloss, prob) in enumerate(results):
            y = panel_top + 52 + i * 38
            cv2.rectangle(frame, (15, y - 18), (15 + int(prob * 280), y + 5),
                          (0, 210, 100) if i == 0 else (50, 110, 70), -1)
            cv2.putText(frame, f"{gloss}  {prob*100:.1f}%", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75 if i == 0 else 0.58,
                        (255, 255, 255), 2 if i == 0 else 1)

    if history:
        cv2.putText(frame, "Recent:", (w - 230, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
        for i, word in enumerate(reversed(history)):
            cv2.putText(frame, word, (w - 230, 103 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 190, 255), 1)


# main
def run_webcam(
    model_path:    str   = "saved_models/asl_citizen_fc_model.pth",
    processed_dir: str   = "asl_citizen_processed",
    top_k:         int   = 3,
    hidden_size:   int   = 150,
    n_layers:      int   = 7,
    dropout:       float = 0.5,
):
    processed_path = Path(processed_dir)

    label_map      = pd.read_csv(processed_path / "label_map.csv")
    label_to_gloss = dict(zip(label_map["label"].astype(int), label_map["gloss"]))
    num_classes    = len(label_to_gloss)

    config_path = processed_path / "config.csv"
    feature_dim = (int(pd.read_csv(config_path).iloc[0]["feature_dim"])
                   if config_path.exists() else FEATURE_DIM)
    print(f"Classes: {num_classes}   Feature dim: {feature_dim}")

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return

    device = (torch.device("mps")  if torch.backends.mps.is_available()  else
              torch.device("cuda") if torch.cuda.is_available()           else
              torch.device("cpu"))
    print(f"Device: {device}")
    model = load_model(model_path, num_classes, hidden_size,
                       n_layers, dropout, feature_dim, device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("\n🎥 Ready! Show a sign to begin.")

    # None entries = frames where no hands were detected
    frame_buffer  = deque(maxlen=WINDOW_SIZE)
    frame_counter = 0
    results       = []
    history       = deque(maxlen=8)
    state         = IDLE
    cooldown_start = 0.0

    with Extractor() as extractor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            display = frame.copy()
            now     = time.time()

            # extract features — tensor is None if no hands detected
            tensor, result = extractor.extract_with_result(frame)
            frame_buffer.append(tensor)
            frame_counter += 1

            if result is not None:
                draw_landmarks(display, result)

            hand_frames = sum(1 for f in frame_buffer if f is not None)
            # extractor returns NONE when we don't see any hands
            hands_now   = result is not None

            # HAND PRESENCE
            if state == COOLDOWN:
                if now - cooldown_start > COOLDOWN_SEC:
                    state = IDLE

            elif state == IDLE:
                if hands_now:
                    state = SIGNING

            elif state == SIGNING:
                if not hands_now:
                    # hands dropped — sign likely ended, force a prediction
                    maybe_predict(frame_buffer, model, label_to_gloss,
                                   top_k, device, results, history, force=True)
                    state          = COOLDOWN
                    cooldown_start = now

                elif frame_counter % STRIDE == 0 and hand_frames >= MIN_HAND_FRAMES:
                    # periodic mid-sign inference — displays only if confident enough
                    emitted = maybe_predict(frame_buffer, model, label_to_gloss,
                                             top_k, device, results, history, force=False)
                    if emitted:
                        state          = COOLDOWN
                        cooldown_start = now

            #  draw 
            _draw_continuous_ui(display, state, results, list(history), hand_frames)
            cv2.imshow("ASL Citizen — Continuous", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                history.clear()
                results = []
            elif key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default="saved_models/asl_citizen_fc_model.pth")
    p.add_argument("--processed-dir", default="asl_citizen_processed")
    p.add_argument("--top-k",         type=int,   default=3)
    p.add_argument("--hidden-size",   type=int,   default=150)
    p.add_argument("--layers",        type=int,   default=7)
    p.add_argument("--dropout",       type=float, default=0.5)
    args = p.parse_args()

    run_webcam(
        model_path=args.model,
        processed_dir=args.processed_dir,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        n_layers=args.layers,
        dropout=args.dropout,
    )