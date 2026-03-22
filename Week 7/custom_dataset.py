from collections import defaultdict
from pathlib import Path
from random import random
from shutil import copy2
import time

import cv2
import pandas as pd
import torch
from dataloader import ImageToTensorPreprocessor


class CustomDatasetCreator:
    def __init__(
        self,
        dataset_name: str = "custom_dataset",
        data_type: str = "video",
        camera_index: int = 0,
        countdown_seconds: float = 1.5,
        start_key: str = "w",
        stop_key: str = "s",
        quit_key: str = "q",
        min_frames_per_clip: int = 2,
    ):
        if data_type != "video":
            raise ValueError("CustomDatasetCreator currently supports only data_type='video'.")

        self.dataset_name = dataset_name
        self.data_type = data_type
        self.countdown_seconds = countdown_seconds
        self.start_key = start_key.lower()
        self.stop_key = stop_key.lower()
        self.quit_key = quit_key.lower()
        self.min_frames_per_clip = min_frames_per_clip

        self.preprocessor = ImageToTensorPreprocessor(
            output_format="landmarks",
            landmark_normalization_method="per-frame-wrist",
            static_image_mode=False,
            draw_on_img=True,
            max_hands=2,
        )

        self.webcam = cv2.VideoCapture(camera_index)
        if not self.webcam.isOpened():
            raise RuntimeError(f"Unable to open camera index {camera_index}.")

        self.src_directory = Path(dataset_name)
        self.src_directory.mkdir(parents=True, exist_ok=True)
        self.label_map_path = self.src_directory / "label_map.csv"

        if not self.label_map_path.exists():
            pd.DataFrame({"gesture_name": [], "label": []}).to_csv(self.label_map_path, index=False)

        label_map = pd.read_csv(self.label_map_path)
        if not label_map.empty:
            label_map = label_map[
                ~label_map["gesture_name"].astype(str).str.match(r"^gesture_\d+$")
            ].reset_index(drop=True)
            label_map["label"] = range(len(label_map))
            label_map.to_csv(self.label_map_path, index=False)

        self.label_to_name = dict(zip(label_map["label"].astype(int), label_map["gesture_name"]))
        self.name_to_label = dict(zip(label_map["gesture_name"], label_map["label"].astype(int)))

    def __del__(self):
        try:
            self.webcam.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    def write_label_map(self):
        output_dict = defaultdict(list)
        for name, label in sorted(self.name_to_label.items(), key=lambda item: item[1]):
            output_dict["gesture_name"].append(name)
            output_dict["label"].append(label)
        pd.DataFrame(output_dict).to_csv(self.label_map_path, index=False)

    def _next_clip_index(self, gesture_dir: Path) -> int:
        existing_clip_ids = []
        for item in gesture_dir.glob("*.pt"):
            try:
                existing_clip_ids.append(int(item.stem))
            except ValueError:
                continue
        return max(existing_clip_ids) + 1 if existing_clip_ids else 0

    def __call__(self, gesture_name: str = "palm_up"):
        gesture_dir = self.src_directory / gesture_name
        gesture_dir.mkdir(parents=True, exist_ok=True)

        if gesture_name not in self.name_to_label:
            label = max(self.name_to_label.values(), default=-1) + 1
            self.name_to_label[gesture_name] = label
            self.write_label_map()
        else:
            label = self.name_to_label[gesture_name]

        csv_path = self.src_directory / f"{label}_{gesture_name}_records.csv"
        clip_index = self._next_clip_index(gesture_dir)
        records: list[dict] = []

        is_recording = False
        is_starting = False
        start_time = 0.0
        current_clip_frames: list[torch.Tensor] = []

        window_name = f"{self.dataset_name} / {gesture_name}"
        print(f"\nPreparing recording loop for: {window_name}")
        print(
            f"[{self.start_key.upper()}] start recording, "
            f"[{self.stop_key.upper()}] stop+save recording, "
            f"[{self.quit_key.upper()}] quit"
        )

        while True:
            ret, frame = self.webcam.read()
            if not ret:
                print("Camera read failed. Ending recording loop.")
                break

            frame_tensor, display_frame = self.preprocessor(frame, return_img=True)
            display_frame = cv2.flip(display_frame, 1)

            key = cv2.waitKey(1) & 0xFF
            key_char = chr(key).lower() if key != 255 else ""

            now = time.time()

            if key_char == self.quit_key:
                if is_recording and len(current_clip_frames) >= self.min_frames_per_clip:
                    clip_tensor = torch.stack(current_clip_frames)
                    torch.save(clip_tensor, gesture_dir / f"{clip_index}.pt")
                    records.append(
                        {
                            "index": clip_index,
                            "label": label,
                            "gesture_name": gesture_name,
                            "num_frames": len(current_clip_frames),
                        }
                    )
                    print(f"Saved clip idx={clip_index} with {len(current_clip_frames)} frames before quitting.")
                    clip_index += 1
                break

            if not is_recording and not is_starting and key_char == self.start_key:
                is_starting = True
                start_time = now
                current_clip_frames = []

            if is_starting:
                elapsed = now - start_time
                countdown_left = max(0.0, self.countdown_seconds - elapsed)
                cv2.putText(
                    display_frame,
                    f"Recording starts in {countdown_left:.1f}s",
                    (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if elapsed >= self.countdown_seconds:
                    is_starting = False
                    is_recording = True
                    print("Recording started...")

            if is_recording:
                if frame_tensor is not None:
                    current_clip_frames.append(frame_tensor)

                cv2.putText(
                    display_frame,
                    f"REC frames={len(current_clip_frames)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                if key_char == self.stop_key:
                    if len(current_clip_frames) >= self.min_frames_per_clip:
                        clip_tensor = torch.stack(current_clip_frames)
                        torch.save(clip_tensor, gesture_dir / f"{clip_index}.pt")
                        records.append(
                            {
                                "index": clip_index,
                                "label": label,
                                "gesture_name": gesture_name,
                                "num_frames": len(current_clip_frames),
                            }
                        )
                        print(f"Saved clip idx={clip_index} with {len(current_clip_frames)} frames.")
                        clip_index += 1
                    else:
                        print(
                            "Discarded short clip with "
                            f"{len(current_clip_frames)} frames (minimum is {self.min_frames_per_clip})."
                        )

                    current_clip_frames = []
                    is_recording = False
                    is_starting = False

            if not is_recording and not is_starting:
                cv2.putText(
                    display_frame,
                    f"Press {self.start_key.upper()} to start, {self.quit_key.upper()} to quit",
                    (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (50, 205, 50),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, display_frame)

        cv2.destroyAllWindows()

        if records:
            df = pd.DataFrame(records)
            if csv_path.exists():
                df.to_csv(csv_path, mode="a", header=False, index=False)
            else:
                df.to_csv(csv_path, index=False)

        print(f"Data added for {gesture_name}: {len(records)} clips")
        print("Recording session finished.")


class CustomDatasetProcessor:
    def __init__(
        self,
        dataset_name: str = "custom_dataset",
        train_val_split: float = 0.8,
        data_type: str = "video",
    ) -> None:
        #if data_type != "video":
            #raise ValueError("")

        self.train_val_split = train_val_split
        self.data_type = data_type
        self.train_index = 0
        self.val_index = 0

        self.src_directory = Path(dataset_name)
        self.label_map_path = self.src_directory / "label_map.csv"
        self.label_map = pd.read_csv(self.label_map_path)
        self.label_to_name = dict(zip(self.label_map["label"].astype(int), self.label_map["gesture_name"]))
        self.name_to_label = dict(zip(self.label_map["gesture_name"], self.label_map["label"].astype(int)))
        self.num_classes = len(self.label_to_name)

        self.tgt_directory = self.src_directory.parent / f"{self.src_directory.name}_processed"
        if self.tgt_directory.exists():
            import shutil

            shutil.rmtree(self.tgt_directory)

        self.tgt_train = self.tgt_directory / "train"
        self.tgt_train.mkdir(parents=True, exist_ok=True)
        self.tgt_val = self.tgt_directory / "val"
        self.tgt_val.mkdir(parents=True, exist_ok=True)

        self.dataset = {"index": [], "partition": [], "label": [], "position": []}

        self._process_dataset()
        print(f"Processed dataset {dataset_name}!")

        pd.DataFrame(self.dataset).to_csv(self.tgt_directory / "gestures.csv", index=False)
        self.label_map.to_csv(self.tgt_directory / "label_map.csv", index=False)

    def _add_record(self, index: int, partition: str, label: int, position: str):
        self.dataset["index"].append(index)
        self.dataset["partition"].append(partition)
        self.dataset["label"].append(label)
        self.dataset["position"].append(position)

    def _process_dataset(self):
        for _, row in self.label_map.iterrows():
            gesture_name = row["gesture_name"]
            label = int(row["label"])

            gesture_dir = self.src_directory / gesture_name
            if not gesture_dir.exists():
                continue

            for example in sorted(gesture_dir.glob("*.pt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
                random_number = random()
                if random_number > self.train_val_split:
                    copy2(example, self.tgt_val / f"{self.val_index}.pt")
                    self._add_record(self.val_index, "val", label, gesture_name)
                    self.val_index += 1
                else:
                    copy2(example, self.tgt_train / f"{self.train_index}.pt")
                    self._add_record(self.train_index, "train", label, gesture_name)
                    self.train_index += 1
            
