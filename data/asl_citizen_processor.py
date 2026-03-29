"""
ASL Citizen Dataset Processor  (v3 — new mediapipe tasks API)

Feature vector per frame: 84 dims
    [left_hand (42) | right_hand (42)]
    Each hand: 21 landmarks × (x, y)

Normalisations:
    1. Position — subtract anchor wrist so location-invariant
    2. Scale    — divide by wrist→middle-MCP distance so signer-invariant

Requires:  mediapipe >= 0.10,  hand_landmarker.task (auto-downloaded)

Run:
    python3 asl_citizen_processor.py --top-n 50
    python3 asl_citizen_processor.py --top-n 50 --max-videos 50
"""

import cv2
import torch
import pandas as pd
import mediapipe as mp
import numpy as np
import urllib.request
import shutil
from pathlib import Path
from tqdm import tqdm
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


WRIST_IDX      = 0
MIDDLE_MCP_IDX = 9   # landmark 9 = middle finger knuckle
FEATURE_DIM    = 84  # 84 numbers per frame

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = Path("hand_landmarker.task")


# normalisation

def _lm_to_xy(landmark_list) -> np.ndarray:
    """Converts a mediapipe landmark list into (21, 2) numpy array."""
    return np.array([[lm.x, lm.y] for lm in landmark_list], dtype=np.float32)


def normalise_frame(left_xy, right_xy) -> torch.Tensor:
    """
    Normalises both hands so the output doesn't depend on where in the frame
    the signer is or how big their hands appear on camera.

    Anchor hand (right preferred, falls back to left) provides:
      - position reference: subtract its wrist from all landmarks
      - scale reference: divide by wrist-to-middle-knuckle distance

    If no hands were detected at all, returns zero tensor.
    """
    # pick whichever hand we'll anchor to
    anchor = None
    if right_xy is not None and not np.allclose(right_xy, 0):
        anchor = right_xy
    elif left_xy is not None and not np.allclose(left_xy, 0):
        anchor = left_xy

    if anchor is None:
        return torch.zeros(FEATURE_DIM, dtype=torch.float32)

    wrist_pos  = anchor[WRIST_IDX]
    middle_mcp = anchor[MIDDLE_MCP_IDX]
    scale = float(np.linalg.norm(middle_mcp - wrist_pos))
    if scale < 1e-6:
        scale = 1.0

    def _norm(xy):
        if xy is None:
            return np.zeros(42, dtype=np.float32)
        return ((xy - wrist_pos) / scale).flatten()

    return torch.tensor(
        np.concatenate([_norm(left_xy), _norm(right_xy)]),
        dtype=torch.float32,
    )


#  extractor

class Extractor:
    """
    Runs mediapipe's HandLandmarker on video frames and gives back
    normalised feature tensors.

    Uses VIDEO mode (not IMAGE mode) so mediapipe can track hands across
    frames rather than re-detecting from scratch each time — much faster.
    Use as a context manager so the landmarker gets cleaned up properly.
    """

    def __init__(self):
        # download the model weights if we don't have them yet
        if not MODEL_PATH.exists():
            print("Downloading hand_landmarker.task …")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Downloaded.")

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker    = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms  = 0 

    def extract_with_result(self, bgr_frame):
        """
        Like extract(), but also returns the raw mediapipe result so the
        caller can draw skeleton overlays on screen.
        Returns (tensor_or_None, result_or_None).
        """
        rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._timestamp_ms += 42

        if not result.hand_landmarks:
            return None, None

        left_xy = right_xy = None
        for hand_lms, handedness in zip(result.hand_landmarks, result.handedness):
            xy    = _lm_to_xy(hand_lms)
            label = handedness[0].category_name
            if label == "Left":
                left_xy = xy
            else:
                right_xy = xy

        return normalise_frame(left_xy, right_xy), result

    def reset_timestamp(self):
        self._landmarker.close()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker   = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def extract(self, bgr_frame) -> torch.Tensor | None:
        """
        Process one BGR frame. Returns a normalised (84,) tensor,
        or None if no hands were detected.
        """
        rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._timestamp_ms += 42 

        if not result.hand_landmarks:
            return None

        left_xy = right_xy = None
        for hand_lms, handedness in zip(result.hand_landmarks, result.handedness):
            xy    = _lm_to_xy(hand_lms)
            label = handedness[0].category_name  # "Left" or "Right"
            if label == "Left":
                left_xy = xy
            else:
                right_xy = xy

        return normalise_frame(left_xy, right_xy)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# processor 

class ASLCitizenProcessor:
    """
    Walks through the ASL Citizen video dataset and converts each clip into
    a (T, 84) PyTorch tensor — one row per frame, 84 hand-landmark features.

    Output layout under asl_citizen_processed/:
        train/ val/ test/   numbered .pt files (one per video)
        glosses.csv         index → partition, label int, gloss string
        label_map.csv       label int ↔ gloss string (for inference)
        config.csv          feature_dim & num_classes (read by trainer)
    """

    def __init__(
        self,
        src_directory: str = "ASL_Citizen",
        top_n_glosses: int = 50,  # only process the most common signs
        min_frames:    int = 10,  # skip clips with fewer valid frames than this
        max_videos:    int = None,  # cap per split
    ):
        self.src_dir    = Path(src_directory)
        self.min_frames = min_frames
        self.max_videos = max_videos
        self.tgt_dir    = self.src_dir.parent / "asl_citizen_processed"
        self.mp4_examples_dir = self.tgt_dir / "mp4_examples"
        self.mp4_examples_dir.mkdir(parents=True, exist_ok=True)
        self.mp4_example_by_gloss = {}

        for split in ["train", "val", "test"]:
            (self.tgt_dir / split).mkdir(parents=True, exist_ok=True)

        #  load the train/val/test split CSVs that come with the dataset 
        print("Loading splits …")
        splits_dir = self.src_dir / "splits"
        parts = []
        for s in ["train", "val", "test"]:
            part = pd.read_csv(splits_dir / f"{s}.csv")
            part.columns = [c.strip().lower().replace(" ", "_") for c in part.columns]
            part.rename(columns={"video_file": "filename"}, inplace=True)
            part["split"] = s
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)

        # drop everything except the top-N most frequent signs 
        counts = df["gloss"].value_counts()
        if top_n_glosses is not None:
            top = counts.head(top_n_glosses).index.tolist()
            df  = df[df["gloss"].isin(top)].copy()
            print(f"Keeping top {top_n_glosses} glosses ({len(df)} videos)")
        else:
            top = counts.index.tolist()

        # sort alphabetically so label integers are stable across runs
        self.gloss_to_label = {g: i for i, g in enumerate(sorted(top))}
        self.label_to_gloss = {i: g for g, i in self.gloss_to_label.items()}
        self.num_classes    = len(self.gloss_to_label)
        print(f"Classes: {self.num_classes}   Feature dims: {FEATURE_DIM}")

        # main processing loop 
        records  = {"index": [], "partition": [], "label": [],
                    "gloss": [], "original_filename": []}
        counters = {"train": 0, "val": 0, "test": 0}

        with Extractor() as extractor:
            for split_name in ["train", "val", "test"]:
                split_df = df[df["split"] == split_name].copy()
                if self.max_videos is not None:
                    split_df = split_df.head(self.max_videos)

                print(f"\nProcessing {split_name} ({len(split_df)} videos) …")
                ok = skipped = 0

                for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
                    filename = row["filename"]
                    gloss    = row["gloss"]
                    label    = self.gloss_to_label[gloss]

                    # try the filename as-is, then with .mp4 appended
                    video_path = self.src_dir / "videos" / filename
                    if not video_path.exists():
                        video_path = self.src_dir / "videos" / (filename + ".mp4")
                    if not video_path.exists():
                        skipped += 1
                        continue

                    # reset mediapipe between videos (timestamp must restart)
                    extractor.reset_timestamp()
                    tensor = self._process_video(video_path, extractor)
                    if tensor is None:
                        skipped += 1
                        continue

                    idx = counters[split_name]
                    torch.save(tensor, self.tgt_dir / split_name / f"{idx}.pt")

                    if gloss not in self.mp4_example_by_gloss:
                        example_name = f"{gloss}.mp4"
                        shutil.copy2(video_path, self.mp4_examples_dir / example_name)
                        self.mp4_example_by_gloss[gloss] = example_name

                    records["index"].append(idx)
                    records["partition"].append(split_name)
                    records["label"].append(label)
                    records["gloss"].append(gloss)
                    records["original_filename"].append(filename)

                    counters[split_name] += 1
                    ok += 1

                print(f"  ✓ {ok} saved   {skipped} skipped")

        # write out metadata
        pd.DataFrame(records).to_csv(
            self.tgt_dir / "glosses.csv", index=False)
        label_map_df = pd.DataFrame(
            list(self.label_to_gloss.items()),
            columns=["label", "gloss"],
        )
        label_map_df["mp4_example"] = label_map_df["gloss"].map(
            lambda gloss: self.mp4_example_by_gloss.get(gloss, "")
        )
        label_map_df.to_csv(self.tgt_dir / "label_map.csv", index=False)
        pd.DataFrame({"feature_dim": [FEATURE_DIM],
                      "num_classes":  [self.num_classes]}).to_csv(
            self.tgt_dir / "config.csv", index=False)

        print(f"\n Done!  →  {self.tgt_dir}")
        print(f"   train={counters['train']}  val={counters['val']}  "
              f"test={counters['test']}")
        print(f"   Feature dim : {FEATURE_DIM}   Classes : {self.num_classes}")

    def _process_video(self, video_path: Path, extractor: Extractor):
        """
        Reads every frame of a video file and runs landmark extraction on it.
        Frames where no hands were detected are dropped entirely.
        Returns a (T, 84) tensor, or None if there weren't enough valid frames.
        """
        cap    = cv2.VideoCapture(str(video_path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            t = extractor.extract(frame)
            if t is not None:
                frames.append(t)
        cap.release()

        if len(frames) < self.min_frames:
            return None
        return torch.stack(frames)   # (T, 84)


# CLI 

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Process ASL Citizen videos into landmark tensors (84-dim)"
    )
    p.add_argument("--src",        default="ASL_Citizen")
    p.add_argument("--top-n",      type=int, default=50,
                   help="Most-frequent glosses to keep (0 = all 2731)")
    p.add_argument("--min-frames", type=int, default=10)
    p.add_argument("--max-videos", type=int, default=None,
                   help="Cap per split — e.g. 50 for a quick smoke-test")
    args = p.parse_args()

    ASLCitizenProcessor(
        src_directory=args.src,
        top_n_glosses=None if args.top_n == 0 else args.top_n,
        min_frames=args.min_frames,
        max_videos=args.max_videos,
    )