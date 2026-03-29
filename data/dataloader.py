import json
import cv2
import numpy as np
from random import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import torch
from shutil import copy

from pathlib import Path
from typing import Union, Callable
import numpy.typing as npt

import pandas as pd
from collections import defaultdict, OrderedDict

# ====================================================================================
# silences mediapipe stuff
import sys
import os
sys.unraisablehook = lambda unraisable: None
# ====================================================================================

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ====================================================================================
# Restore printing so that tqdm displays correctly
#os.dup2(_old_stdout, 1)
#os.dup2(_old_stderr, 2)
#os.close(_devnull)
#os.close(_old_stdout)
#os.close(_old_stderr)
# ====================================================================================


#import sys
#sys.path.insert(0, '..')
#from sys import getsizeof



most_letters = "ABCDEFGHIKLMNOPQRSTUVWXY"
LETTERS_FEATURE_DIM = 84

def letter_to_label(letter: str) -> int:
    """converts ASL letter (A-Z, excluding J and Z) to integer label 0-24. """
    letter = letter.upper()
    if letter not in most_letters:
        raise ValueError(f"Invalid letter: {letter}")
    return most_letters.index(letter)

def label_to_letter(label: int) -> str:
    return most_letters[label]


class LettersDatasetProcessor:
    def __init__(
        self,
        src_directory="asl_letters",
        train_val_split=0.8,
        filter_to_landmarkable=True,
        included_letters: list[str] = None,
        to_video: bool = False,
        video_seconds: int = 4,
        video_fps: int = 24,
        save_landmark_examples: bool = False,
        max_examples_per_letter: int = 1000
    ) -> None:
        self.train_val_split = train_val_split
        self.excluded_letters = ["del", "space", "J", "Z", "nothing"]
        self.to_video = to_video
        self.video_seconds = video_seconds
        self.video_fps = video_fps
        self.video_frames = self.video_seconds * self.video_fps
        self.max_examples_per_letter = max_examples_per_letter

        if included_letters is not None:
            self.included_letters = included_letters
        else: 
            self.included_letters = list(most_letters)
        
        self.filter_to_landmarkable = filter_to_landmarkable
        self.landmark_checker = ImageToTensorPreprocessor(
            output_format="landmarks",
            draw_on_img=True,
            max_hands=2,
            min_detection_confidence=0.25
        )

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.src_directory = Path(src_directory)
        self.src_train_val_dir = self.src_directory / "asl_alphabet_train/asl_alphabet_train"
        self.src_test_dir = self.src_directory / "asl_alphabet_test/asl_alphabet_test"

        self.tgt_directory = self.src_directory.parent / f"{self.src_directory.name}_processed"

        # ------------------------------------------------------
        # make directories to store image with mediapipe overlay
        self.save_landmark_examples = save_landmark_examples
        if self.save_landmark_examples:
            self.preview_train = self.tgt_directory / "train_landmarks" 
            self.preview_train.mkdir(parents=True, exist_ok=True)
            self.preview_val = self.tgt_directory / "val_landmarks" 
            self.preview_val.mkdir(parents=True, exist_ok=True)
        # ------------------------------------------------------

        self.tgt_train = self.tgt_directory / "train"
        self.tgt_train.mkdir(parents=True, exist_ok=True)

        self.tgt_val = self.tgt_directory / "val"
        self.tgt_val.mkdir(parents=True, exist_ok=True)

        self.tgt_test = self.tgt_directory / "test"
        self.tgt_test.mkdir(parents=True, exist_ok=True)

        self.mp4_examples_dir = self.tgt_directory / "mp4_examples"
        self.mp4_examples_dir.mkdir(parents=True, exist_ok=True)
        self.mp4_example_by_gloss = {}

        if self.to_video:
            self.dataset = {
                "index": [],
                "partition": [],
                "label": [],
                "gloss": [],
                "original_filename": [],
            }
        else:
            self.dataset = {"index": [], "partition": [], "label": [], "letter": []}

        self._process_dataset()
        print('processed dataset!')

        df = pd.DataFrame(self.dataset)
        label_set = {
            letter_to_label(letter)
            for letter in self.included_letters
            if letter in most_letters
        }
        label_map_df = pd.DataFrame(
            {
                "label": sorted(label_set),
                "gloss": [label_to_letter(label) for label in sorted(label_set)],
            }
        )
        label_map_df["mp4_example"] = label_map_df["gloss"].map(
            lambda gloss: self.mp4_example_by_gloss.get(gloss, "")
        )

        if self.to_video:
            df.to_csv(self.tgt_directory / "glosses.csv", index=False)
            label_map_df.to_csv(self.tgt_directory / "label_map.csv", index=False)

            pd.DataFrame(
                {
                    "feature_dim": [LETTERS_FEATURE_DIM],
                    "num_classes": [len(label_map_df)],
                }
            ).to_csv(self.tgt_directory / "config.csv", index=False)
        else:
            df.to_csv(self.tgt_directory / "glosses.csv", index=False)
            label_map_df.to_csv(self.tgt_directory / "label_map.csv", index=False)
    
    def _make_video_tensor(self, landmark_tensor: torch.Tensor) -> torch.Tensor:
        """Convert one static (84,) landmark tensor into a fixed-length (T,84) sequence."""
        if self.video_frames <= 0:
            raise ValueError("video_frames must be > 0")
        return landmark_tensor.unsqueeze(0).repeat(self.video_frames, 1)

    def _save_mp4_example(self, gloss: str, image_bgr: npt.ArrayLike):
        """Save a single 4-second mp4 preview clip per gloss based on a static image."""
        if gloss in self.mp4_example_by_gloss:
            return
        if image_bgr is None:
            return

        output_name = f"{gloss}.mp4"
        output_path = self.mp4_examples_dir / output_name
        height, width = image_bgr.shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.video_fps,
            (width, height),
        )
        for _ in range(self.video_frames):
            writer.write(image_bgr)
        writer.release()

        self.mp4_example_by_gloss[gloss] = output_name

    def _add_record(self, index: int, partition: str, label: int, letter: str, original_filename: str = None):
        self.dataset["index"].append(index)
        self.dataset["partition"].append(partition)
        self.dataset["label"].append(label)
        if self.to_video:
            self.dataset["gloss"].append(letter)
            self.dataset["original_filename"].append(original_filename if original_filename is not None else "")
        else:
            self.dataset["letter"].append(letter)

    def _process_dataset(self):
        # add train and validation data: up to max_examples_per_letter observations for each letter

        i = 0
        for letter in self.src_train_val_dir.iterdir():
            if letter.name in self.excluded_letters:
                continue
            if letter.name == ".DS_Store":
                continue
            if len(self.included_letters) != 0:
                if letter.name not in self.included_letters:
                    continue
            #else:
                #print(f"this is {letter.name}")
            
            examples_count = 0
            for example in letter.iterdir():
                if examples_count >= self.max_examples_per_letter:
                    break
                
                img = cv2.imread(Path.cwd() / example)
                if img is None:
                    continue

                img_landmarks = None
                if self.to_video or self.filter_to_landmarkable:
                    img_landmarks = self.landmark_checker(img)
                    if img_landmarks is None:
                        # Skip images where landmarks cannot be detected
                        continue

                self._save_mp4_example(letter.name, img)
                examples_count += 1
                
                random_number = random()
                if random_number > self.train_val_split:
                    if self.to_video:
                        video_tensor = self._make_video_tensor(img_landmarks)
                        torch.save(video_tensor, self.tgt_val / f"{self.val_index}.pt")
                        self._add_record(
                            self.val_index,
                            "val",
                            letter_to_label(letter.name),
                            letter.name,
                            original_filename=example.name,
                        )
                    else:
                        copy(Path.cwd() / example, self.tgt_val / f"{self.val_index}.jpg")
                        self._add_record(self.val_index, "val", letter_to_label(letter.name), letter.name)
                    if self.save_landmark_examples and self.filter_to_landmarkable and not self.to_video:
                        cv2.imwrite(self.preview_val / f"val_{self.val_index}.jpg", img)
                    self.val_index += 1
                else:
                    if self.to_video:
                        video_tensor = self._make_video_tensor(img_landmarks)
                        torch.save(video_tensor, self.tgt_train / f"{self.train_index}.pt")
                        self._add_record(
                            self.train_index,
                            "train",
                            letter_to_label(letter.name),
                            letter.name,
                            original_filename=example.name,
                        )
                    else:
                        copy(Path.cwd() / example, self.tgt_train / f"{self.train_index}.jpg")
                        self._add_record(self.train_index, "train", letter_to_label(letter.name), letter.name)
                    if self.save_landmark_examples and self.filter_to_landmarkable and not self.to_video:
                        cv2.imwrite(self.preview_train / f"train_{self.train_index}.jpg", img)
                    self.train_index += 1
            
        # add test data: one observation each
        for letter in self.included_letters:
            test_img_path = self.src_test_dir / f"{letter}_test.jpg"
            test_img = cv2.imread(test_img_path)
            self._save_mp4_example(letter, test_img)
            if self.to_video:
                img_landmarks = self.landmark_checker(test_img)
                if img_landmarks is None:
                    continue
                video_tensor = self._make_video_tensor(img_landmarks)
                torch.save(video_tensor, self.tgt_test / f"{self.test_index}.pt")
                self._add_record(
                    self.test_index,
                    "test",
                    letter_to_label(letter),
                    letter,
                    original_filename=test_img_path.name,
                )
            else:
                copy(test_img_path, self.tgt_test / f"{self.test_index}.jpg")
                self._add_record(self.test_index, "test", letter_to_label(letter), letter)
            self.test_index += 1


class VideoDatasetProcessor:
    def __init__(self, src_directory="asl_glosses", train_val_split = 0.8, filter_to_landmarkable=False, top_n: int = 5, excluded_glosses: list[str] = None, selected_glosses: list[str] = None, save_landmark_examples: bool = False) -> None:
        """
        Dennis: To use this with other possible video dataset, create a copy of this with a modified:
        - self.src_directory
        - loop over samples.json or equivalent (possibly iterate over files like in LettersDatasetProcessor)

        I could likely implement this more elegantly with inheritance
        
        If selected_glosses is provided, only those glosses will be used (top_n is ignored).
        """
        self.train_val_split = train_val_split

        if excluded_glosses is None:
            self.excluded_glosses = []
        else:
            self.excluded_glosses = excluded_glosses
        
        self.selected_glosses = selected_glosses


        self.filter_to_landmarkable = filter_to_landmarkable
        self.landmark_checker = ImageToTensorPreprocessor(
            output_format="landmarks",
            static_image_mode=False,
            draw_on_img=True,
            min_detection_confidence=0.25,
            min_tracking_confidence=0.5
        )
        
        # Separate preprocessor for tensor extraction (no drawing)
        self.tensor_extractor = ImageToTensorPreprocessor(
            output_format="landmarks",
            landmark_normalization_method="per-frame-wrist",
            static_image_mode=False,
            draw_on_img=False,
            max_hands=2,
            min_detection_confidence=0.25,
            min_tracking_confidence=0.5
        )

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        ########################################################################################
        ########################################################################################
        # Modify this section for alternative dataset compatability (as applicable)

        self.src_directory = Path(src_directory)

        samples = []
        file_path = self.src_directory / "samples.json"
        with open(file_path) as ipf:
            samples_json = json.load(ipf)["samples"]
        for sample in samples_json:
            samples.append({
                "id": sample["_id"]["$oid"],
                "dataset_id": sample["_dataset_id"]["$oid"],
                "filepath": str((self.src_directory / sample["filepath"]).absolute()),
                "gloss": sample["gloss"]["label"],
                "bounding_box": sample["bounding_box"]["detections"][0]["bounding_box"],
                "metadata": sample["metadata"],
                })

        glosses_grouped = defaultdict(list)
        for sample in samples:
            glosses_grouped[sample["gloss"]].append(sample)

        ########################################################################################
        ########################################################################################

        for k in self.excluded_glosses:
            try:
                glosses_grouped.pop(k, None)
            except KeyError:
                pass

        # Use selected_glosses if provided, otherwise use top_n
        if self.selected_glosses is not None:
            # Filter to only selected glosses
            self.glosses = OrderedDict(
                (gloss, examples) for gloss, examples in glosses_grouped.items()
                if gloss in self.selected_glosses
            )
            self.top_n = len(self.glosses)
        else:
            self.top_n = len(glosses_grouped) if top_n is None else top_n
            self.glosses = dict(
                sorted(glosses_grouped.items(), key=lambda item: len(item[1]), reverse=True)[:self.top_n]
            )
            self.glosses = OrderedDict(self.glosses.items())

        for k,v in list(self.glosses.items()):
            print(f"n occurances of {k}: {len(v)}")

        self.gloss_to_label = lambda gloss: list(self.glosses.keys()).index(gloss)

        self.tgt_directory = self.src_directory.parent / f"{self.src_directory.name}_processed"

        # ------------------------------------------------------
        # make directories to store image with mediapipe overlay
        self.save_landmark_examples = save_landmark_examples
        if self.save_landmark_examples:
            self.preview_train = self.tgt_directory / "train_landmarks" 
            self.preview_train.mkdir(parents=True, exist_ok=True)
            self.preview_val = self.tgt_directory / "val_landmarks" 
            self.preview_val.mkdir(parents=True, exist_ok=True)
        # ------------------------------------------------------

        self.tgt_train = self.tgt_directory / "train"
        self.tgt_train.mkdir(parents=True, exist_ok=True)

        self.tgt_val = self.tgt_directory / "val"
        self.tgt_val.mkdir(parents=True, exist_ok=True)

        self.tgt_test = self.tgt_directory / "test"
        self.tgt_test.mkdir(parents=True, exist_ok=True)

        self.mp4_examples_dir = self.tgt_directory / "mp4_examples"
        self.mp4_examples_dir.mkdir(parents=True, exist_ok=True)
        self.mp4_example_by_gloss = {}

        self.dataset = {"index": [], "partition": [], "label": [], "gloss": []}

        self._process_dataset()
        print('processed dataset!')

        df = pd.DataFrame(self.dataset)
        df.to_csv(self.tgt_directory / "glosses.csv", index=False)

        label_map_df = pd.DataFrame(
            {
                "label": list(range(len(self.glosses))),
                "gloss": list(self.glosses.keys()),
            }
        )
        label_map_df["mp4_example"] = label_map_df["gloss"].map(
            lambda gloss: self.mp4_example_by_gloss.get(gloss, "")
        )
        label_map_df.to_csv(self.tgt_directory / "label_map.csv", index=False)
        
        pd.DataFrame(
            {
                "feature_dim": [84],
                "num_classes": [len(self.glosses)],
            }
        ).to_csv(self.tgt_directory / "config.csv", index=False)
    
    def _process_video_to_tensor(self, video_path: Path) -> torch.Tensor:
        """Load video and extract landmarks as (T, 84) tensor."""
        video_reader = cv2.VideoCapture(str(video_path))
        is_first_frame = True
        first_frame_landmark_tensor = torch.zeros(84, dtype=torch.float32)
        frames = []
        
        while True:
            ret, frame = video_reader.read()
            if not ret:
                break
            
            current_frame = self.tensor_extractor(frame, first_frame_landmark_tensor=first_frame_landmark_tensor)
            if current_frame is not None:
                frames.append(current_frame)
                if is_first_frame:
                    first_frame_landmark_tensor = current_frame.detach().clone()
                    is_first_frame = False
        
        video_reader.release()
        
        if len(frames) == 0:
            return None
        return torch.stack(frames)
    
    def _add_record(self, index: int, partition: str, label: int, gloss: str):
        self.dataset["index"].append(index)
        self.dataset["partition"].append(partition)
        self.dataset["label"].append(label)
        self.dataset["gloss"].append(gloss)

    def _process_dataset(self):
        temp_output_path = self.tgt_directory / f"temp_annotated.mp4"
        for gloss, examples in self.glosses.items():
            if gloss in self.excluded_glosses:
                continue

            for example in examples:
                input_path = Path.cwd() / example["filepath"]

                if self.filter_to_landmarkable:
                    is_landmarkable = self.landmark_checker.draw_hand_landmarks_video(
                        input_path, temp_output_path
                    )
                    
                    if not is_landmarkable:
                        continue

                if gloss not in self.mp4_example_by_gloss and input_path.exists():
                    output_name = f"{gloss}.mp4"
                    copy(input_path, self.mp4_examples_dir / output_name)
                    self.mp4_example_by_gloss[gloss] = output_name
                
                # Extract landmarks and save as .pt tensor
                # Defer train/val splitting to dataloader stage for proper stratification.
                tensor = self._process_video_to_tensor(input_path)
                if tensor is None:
                    continue
                
                torch.save(tensor, self.tgt_train / f"{self.train_index}.pt")
                self._add_record(self.train_index, "train", self.gloss_to_label(gloss), gloss)
                if self.save_landmark_examples and self.filter_to_landmarkable:
                    temp_output_path.rename(self.preview_train / f"train_{self.train_index}.mp4")
                self.train_index += 1

            if temp_output_path.exists():
                temp_output_path.unlink()

def landmarks_to_tensor(left_hand_coords: list[(int, float, float)] = None, right_hand_coords: list[(int, float, float)] = None) -> torch.Tensor:
    """
    index 0-41: left hand coordinates (x0, y0, x1, y1, ..., x20, y20)
    index 42-83: right hand coordinates (x21, y21, x22, y22, ..., x41, y41)
    """
    output_tensor = torch.zeros(42*2, dtype=torch.float32)
    lh_coords = []
    if left_hand_coords is not None:
        for i,x,y in left_hand_coords:
            lh_coords.extend([x,y])
        output_tensor[:42] += torch.tensor(lh_coords, dtype=torch.float32)

    rh_coords = []
    if right_hand_coords is not None:
        for i,x,y in right_hand_coords:
            rh_coords.extend([x,y])
        output_tensor[42:] += torch.tensor(rh_coords, dtype=torch.float32)

    return output_tensor
                
class ImageToTensorPreprocessor():
    """
    landmark preprocessor: takes the original tensor and an optional normalizing tensor, and returns a transformed landmark tensor

    returns tensor representation of input image

    if draw_on_img==True, modifies img directly

    if `max_hands==1`, uses right hand, or left hand flipped over vertical axis (tensor shape: 42)

    if `max_hands==2`, uses right hand and left hand (tensor shape: 84)
    """
    def __init__(self,
                output_format: "image|landmarks" = "image",
                image_preprocessor: Callable = None,
                landmark_normalization_method: "per-frame-wrist|first-frame-wrist|None" = "per-frame-wrist",
                static_image_mode=True,
                draw_on_img=False,
                return_img=False,
                max_hands=1,
                min_detection_confidence=0.25, 
                min_tracking_confidence=0.5):
        """
        if static_image_mode=False, uses tracking optimization for sequential video frames
        """
        import urllib.request

        # download hand_landmarker.task if not already installed
        model_path = Path("hand_landmarker.task")
        task_object_path = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        if not model_path.exists():
            print("Downloading hand_landmarker.task...")
            urllib.request.urlretrieve(task_object_path, model_path)
            print("Download complete!")

        self.output_format = output_format
        self.results = None
        self.processing_mode = "static" if static_image_mode else "video"
        self.video = self.processing_mode == "video"
        self.draw_on_img = draw_on_img
        self.return_img = return_img

        self.max_hands = max_hands

        # Configure hand landmarker using the new tasks API
        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO if not static_image_mode else vision.RunningMode.IMAGE,
            num_hands=2,  # Detect up to 2 hands, will filter to right hand later
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0

        self.landmark_normalization_method = landmark_normalization_method

        if image_preprocessor is not None:
            self.image_preprocessor = image_preprocessor
        else:
            self.image_preprocessor = lambda image: torch.tensor(image, dtype=torch.float32)
            
    def normalize_to_reference(self, landmark_tensor, norm_tensor_lh=None, norm_tensor_rh=None, subtract_from_wrist: bool = True): 
        """
        `norm_tensor_lh` and `norm_tensor_rh` are assumed to be shape 2

        Use this to implement either per-frame wrist normalization, or first-frame wrist normalization (for invariance to translation within frame)
        """
        if norm_tensor_lh is not None:
            landmark_tensor[:42] -= norm_tensor_lh.repeat(21)
            if not subtract_from_wrist:
                landmark_tensor[0:2] += norm_tensor_lh

        if norm_tensor_rh is not None:
            landmark_tensor[42:84] -= norm_tensor_rh.repeat(21)
            if not subtract_from_wrist:
                landmark_tensor[42:44] += norm_tensor_rh

        return landmark_tensor
    
    def normalize_scale(self, landmark_tensor):
        """
        Normalize landmarks by dividing by the euclidean distance between extreme hand landmarks (for invariance to scale)
        """
        if torch.sum(landmark_tensor[:42]) != 0:
            lh_x = landmark_tensor[0:42:2]
            lh_y = landmark_tensor[1:42:2]
            lh_min_x, lh_max_x = torch.min(lh_x), torch.max(lh_x)
            lh_min_y, lh_max_y = torch.min(lh_y), torch.max(lh_y)
            lh_distance = torch.sqrt((lh_max_x - lh_min_x)**2 + (lh_max_y - lh_min_y)**2)
            if lh_distance > 0:
                landmark_tensor[:42] /= lh_distance
        
        if torch.sum(landmark_tensor[42:]) != 0:
            rh_x = landmark_tensor[42:84:2]
            rh_y = landmark_tensor[43:84:2]
            rh_min_x, rh_max_x = torch.min(rh_x), torch.max(rh_x)
            rh_min_y, rh_max_y = torch.min(rh_y), torch.max(rh_y)
            rh_distance = torch.sqrt((rh_max_x - rh_min_x)**2 + (rh_max_y - rh_min_y)**2)
            if rh_distance > 0:
                landmark_tensor[42:] /= rh_distance
        
        return landmark_tensor

    def flip_landmarks_horizontally(self, landmark_list: list[(float,float)]):
        """assumes that coordinates are normalized to 0-1"""
        return [(i, 1-x, y) for i,x,y in landmark_list]


    def find_hand_landmarks(self, image):
        """Detect hand landmarks using tasks API"""
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to mediapipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        
        # Detect hands
        if self.processing_mode == "video":
            self.results = self.hand_landmarker.detect_for_video(mp_image, self.timestamp_ms)
            #self.timestamp_ms += 33  # approx 30 fps
            self.timestamp_ms += 42  # approx 23.976 fps
        else:
            self.results = self.hand_landmarker.detect(mp_image)

        hand_landmarks = []
        if self.results.hand_landmarks and self.results.handedness:
            for hand_num in range(len(self.results.hand_landmarks)):
                hand = self.results.hand_landmarks[hand_num]
                handedness = self.results.handedness[hand_num][0]
                landmark_list = []
                for id, landmark in enumerate(hand):
                    center_x, center_y = float(landmark.x), float(landmark.y)
                    landmark_list.append([id, center_x, center_y])
                
                hand_landmarks.append((handedness.category_name, landmark_list))

        if len(hand_landmarks) == 0:
            return []

        left_hand = None
        right_hand = None
        for handedness_label, landmarks in hand_landmarks:
            if handedness_label == "Left":
                left_hand = landmarks
            elif handedness_label == "Right":
                right_hand = landmarks

        if self.max_hands == 2:
            result = []
            if left_hand is not None:
                result.append(("Left", left_hand))
            if right_hand is not None:
                result.append(("Right", right_hand))
            return result
        elif self.max_hands == 1:
            if right_hand is not None:
                return [("Right", right_hand)]
            elif left_hand is not None:
                return [("Right", self.flip_landmarks_horizontally(left_hand))]
            else:
                return []

        return []


    def draw_hand_landmarks(self, image, text: str = None):
        """
        directly modifies `image` by superimposing landmark coordinates and right/left hand labels
        """
        #image = np.copy(image)
        if self.results.hand_landmarks:
            mp_drawing = mp.tasks.vision.drawing_utils
            mp_drawing_styles = mp.tasks.vision.drawing_styles


            #def draw_landmarks_on_image(rgb_image, detection_result):
            hand_landmarks_list = self.results.hand_landmarks
            handedness_list = self.results.handedness

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                handedness = handedness_list[idx]

                # Draw the hand landmarks.
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Get the top left corner of the detected hand's bounding box.
                MARGIN = 10  # pixels
                height, width, _ = image.shape
                x_coordinates = [landmark.x for landmark in hand_landmarks]
                y_coordinates = [landmark.y for landmark in hand_landmarks]
                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height) - MARGIN
                if text is None:
                    text = f"{handedness[0].category_name}"

                # Draw handedness (left or right hand) on the image.
                FONT_SIZE = 1
                FONT_THICKNESS = 1
                HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
                cv2.putText(image, text,
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

            #annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
            #cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            #for hand_landmarks in self.results.hand_landmarks:
                #hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                #hand_landmarks_proto.landmark.extend([
                    #landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    #for landmark in hand_landmarks
                #])
                #solutions.drawing_utils.draw_landmarks(
                    #image,
                    #hand_landmarks_proto,
                    #solutions.hands.HAND_CONNECTIONS,
                    #solutions.drawing_styles.get_default_hand_landmarks_style(),
                    #solutions.drawing_styles.get_default_hand_connections_style()
                #)
        return image

    def draw_hand_landmarks_video(self, input_path: Path, output_path: Path) -> bool:
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        video_reader = cv2.VideoCapture(str(input_path))
        
        if not video_reader.isOpened():
            print(f"Warning: Could not open video {input_path}")
            return False, 0
        
        fps = int(video_reader.get(cv2.CAP_PROP_FPS))
        width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frames_with_landmarks = 0
        annotated_frames = []

        
        while True:
            ret, original_frame = video_reader.read()
            if not ret:
                break
            
            landmark_tensor, annotated_frame = self.__call__(original_frame, return_img=True)
            
            if landmark_tensor is not None:
                frames_with_landmarks += 1
                annotated_frames.append(annotated_frame)
            else:
                annotated_frames.append(original_frame)
        
        video_reader.release()
        
        for frame in annotated_frames:
            video_writer.write(frame)
        
        video_writer.release()
        
        return frames_with_landmarks > 0

    def __del__(self):
        try:
            self.hand_landmarker.close()
            del self.hand_landmarker
        except Exception:
            # 'NoneType' object is not callable
            # some sort of underlying MediaPipe unraisable exception thing
            pass

    def image_to_hand_landmarks(self, image) -> list:
        landmarks = self.find_hand_landmarks(image)
        return landmarks

    def __call__(self, image: npt.ArrayLike, first_frame_landmark_tensor: torch.Tensor = None, return_img: bool = None) -> torch.Tensor:
        #landmarks = self.detector.find_position(image)
        if image is None:
            raise ValueError("Input image is None. Cannot process.")
        if return_img is None:
            return_img = False
            
        #image = cv2.flip(image, 1)
        if self.output_format == "image":
            tensor = self.image_preprocessor(image)
            return tensor
        else:
            # image = self.image_preprocessor  
            hand_landmarks = self.image_to_hand_landmarks(image)
            if len(hand_landmarks) == 0:
                # no hand detected
                if self.return_img or return_img:
                    return None, image
                else:
                    return None

            left_hand_coords = None
            right_hand_coords = None

            hands_dict = {hand_name: coords for hand_name, coords in hand_landmarks}
            if "Right" in hands_dict.keys():
                right_hand_coords = hands_dict["Right"]
            if "Left" in hands_dict.keys():
                left_hand_coords = hands_dict["Left"]

            #right_hand_coords = hand_landmarks[0][1]
            landmark_tensor = landmarks_to_tensor(left_hand_coords=left_hand_coords, right_hand_coords=right_hand_coords)

            if self.landmark_normalization_method is None:
                pass
            elif self.landmark_normalization_method == "per-frame-wrist":
                # Normalize both hands to their respective wrists
                norm_tensor_lh = landmark_tensor[0:2] if torch.sum(landmark_tensor[:42]) != 0 else None
                norm_tensor_rh = landmark_tensor[42:44] if torch.sum(landmark_tensor[42:]) != 0 else None
                landmark_tensor = self.normalize_to_reference(landmark_tensor, norm_tensor_lh=norm_tensor_lh, norm_tensor_rh=norm_tensor_rh, subtract_from_wrist=False)
            elif self.landmark_normalization_method == "first-frame-wrist" and first_frame_landmark_tensor is not None:
                # Normalize both hands to first frame wrists
                norm_tensor_lh = first_frame_landmark_tensor[0:2] if torch.sum(first_frame_landmark_tensor[:42]) != 0 else None
                norm_tensor_rh = first_frame_landmark_tensor[42:44] if torch.sum(first_frame_landmark_tensor[42:]) != 0 else None
                landmark_tensor = self.normalize_to_reference(landmark_tensor, norm_tensor_lh=norm_tensor_lh, norm_tensor_rh=norm_tensor_rh)
            
            # Normalize by range (euclidean distance between extreme landmarks)
            landmark_tensor = self.normalize_scale(landmark_tensor)

            if self.draw_on_img:
                image = self.draw_hand_landmarks(image)

            if self.return_img or return_img:
                return landmark_tensor, image
            else:
                return landmark_tensor



# This class is used for both image and video dataloaders, with some code to avoid pickling errors
class GestureDataset(Dataset):
    def __getstate__(self):
        """
        MediaPipe object are unable to be pickled for caching, so prior to saving as a `.pkl` file, this:
        - saves the ImageToTensorPreprocessor state, and 
        - removes it from the Dataset object

        It is then readded when loading the `.pkl` file
        """
        
        state = self.__dict__.copy()
        if isinstance(self.preprocessor, ImageToTensorPreprocessor):
            state['_preprocessor_config'] = {
                'output_format': self.preprocessor.output_format,
                'landmark_normalization_method': self.preprocessor.landmark_normalization_method,
                'draw_on_img': self.preprocessor.draw_on_img,
                'return_img': self.preprocessor.return_img,
                'processing_mode': self.preprocessor.processing_mode,
            }
        # Remove the unpicklable preprocessor
        del state['preprocessor']
        return state

    def __setstate__(self, state):
        """
        This instantiates the ImageToTensorPreprocessor based on the config saved in the pickle file
        """
        if '_preprocessor_config' in state:
            config = state.pop('_preprocessor_config')
            state['preprocessor'] = ImageToTensorPreprocessor(
                output_format=config['output_format'],
                landmark_normalization_method=config['landmark_normalization_method'],
                static_image_mode=(config['processing_mode'] != 'video'),
                draw_on_img=config['draw_on_img'],
                return_img=config['return_img'],
            )
        self.__dict__.update(state)







class ImageDataset(GestureDataset):
    def __init__(
            self,
            directory: Union[str, Path],
            partition: str = "train",
            preprocessor: Callable[[npt.ArrayLike], torch.Tensor] = None,
            indices: list[int] = None,
        ):
            self.partition = partition
            if partition not in ("train", "test", "val"):
                raise ValueError(f"Invalid partition specified - {partition}")
            self.directory = Path(directory)
            self.img_directory = self.directory / partition
            metadata = pd.read_csv(self.directory / "glosses.csv")
            
            if indices is not None:
                # Use provided indices (for stratified split)
                self.metadata = metadata.iloc[indices].reset_index(drop=True)
            else:
                # Use partition column
                self.metadata = metadata[metadata["partition"] == partition]
                self.metadata.reset_index(drop=True, inplace=True)

            if preprocessor is not None:
                self.preprocessor = preprocessor
            else:
                self.preprocessor = ImageToTensorPreprocessor()


    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        row = self.metadata.iloc[index]
        label = row["label"]
        letter = row["letter"]
        file_index = row["index"]
        image_path = str(self.img_directory / f"{file_index}.jpg")
        
        image = cv2.imread(image_path)
        
        if image is None:
            raise RuntimeError(f"Failed to load image at {image_path}. File may be missing or corrupted.")

        #if self.preprocessor.draw_on_img:
            #landmark_tensor, annotated_image = self.preprocessor(image)
        #else:
        landmark_tensor = self.preprocessor(image)

        if landmark_tensor is None:
            raise RuntimeError(
                f"Failed to detect hand landmarks at index {index} (letter: {letter}, path: {image_path}). "
                f"Re-run the dataset processor with filter_to_landmarkable=True."
            )

        return landmark_tensor, label


class VideoDataset(GestureDataset):
    def __init__(
            self,
            directory: Union[str, Path],
            partition: str = "train",
            preprocessor: Callable[[npt.ArrayLike], torch.Tensor] = None,
            indices: list[int] = None,
        ):
            print(f"initializing {partition} dataloader...")
            self.partition = partition
            if partition not in ("train", "test", "val"):
                raise ValueError(f"Invalid partition specified - {partition}")
            self.directory = Path(directory)
            self.img_directory = self.directory / partition
            metadata = pd.read_csv(self.directory / "glosses.csv")
            
            if indices is not None:
                # Use provided indices (for stratified split)
                self.metadata = metadata.iloc[indices].reset_index(drop=True)
            else:
                # Use partition column
                self.metadata = metadata[metadata["partition"] == partition]
                self.metadata.reset_index(drop=True, inplace=True)

            if preprocessor is not None:
                self.preprocessor = preprocessor
            else:
                self.preprocessor = ImageToTensorPreprocessor(
                    output_format="landmarks",
                    landmark_normalization_method="per-frame-wrist",
                    static_image_mode=False, 
                    draw_on_img=False,
                    max_hands=2)
                
            self.landmark_tensors, self.labels = [], []
            self.__save_videos_to_memory()
            print(f"finished initializing {partition} dataloader!")

    def __save_videos_to_memory(self):
        for index, row in self.metadata.iterrows():
            label = row["label"]
            self.landmark_tensors.append(self.__load_video(index))
            self.labels.append(label)

    def __load_video(self, index) -> tuple[torch.Tensor, int]:
        row = self.metadata.iloc[index]
        file_index = row["index"]
        # Load pre-processed tensor from .pt file
        tensor = torch.load(self.img_directory / f"{file_index}.pt")
        return tensor
    
    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        #label = self.metadata.iloc[index]["label"]
        #return self.__load_video(index), label
        return self.landmark_tensors[index], self.labels[index]






def get_dataloader(
    dataset_name: str,
    partition: str,
    as_landmarks: bool = False,
    batch_size: int = 1,
    num_workers = 0,
    shuffle: bool = None,
    train_val_split: float = 0.8,
    random_state: int = 42,
) -> DataLoader:
    """
    dataset: "asl_letters" | "asl_letters_small" | "asl_glosses"
    partition: "train" | "val"

    as_landmarks: if true, load data as landmark tensors, otherwise load as image tensors
    
    Performs stratified train/val split based on labels.
    """

    assert (partition == "train") or (partition == "val")

    preprocessor = ImageToTensorPreprocessor(
        output_format="landmarks" if as_landmarks else "image",
        image_preprocessor=None,
        max_hands=2,
        #landmark_normalization_method=None
        landmark_normalization_method="per-frame-wrist"
        )

    # Determine if dataset_name is letters or glosses based
    if "letters" in dataset_name:
        # Image dataset (letters)
        if dataset_name.endswith("_processed"):
            directory = dataset_name
        else:
            directory = f"{dataset_name}_processed"
        metadata = pd.read_csv(Path(directory) / "glosses.csv")
        # Only use samples originally marked as train (exclude test)
        train_val_metadata = metadata[metadata["partition"] != "test"]
        
        if len(train_val_metadata) > 0:
            labels = train_val_metadata["label"].values
            indices = train_val_metadata.index.values
            
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_val_split, random_state=random_state)
            train_idx, val_idx = next(sss.split(indices, labels))
            
            if partition == "train":
                selected_indices = indices[train_idx]
            else:
                selected_indices = indices[val_idx]
        else:
            selected_indices = train_val_metadata.index.values
        
        dataset = ImageDataset(
            directory=directory,
            partition="train",
            preprocessor=preprocessor,
            indices=list(selected_indices)
        )

    elif "glosses" in dataset_name or "asl_glosses" in dataset_name:
        # Video dataset (glosses)
        if dataset_name.endswith("_processed"):
            directory = dataset_name
        else:
            directory = f"{dataset_name}_processed"
        metadata = pd.read_csv(Path(directory) / "glosses.csv")
        # Only use samples originally marked as train (exclude test)
        train_val_metadata = metadata[metadata["partition"] != "test"]
        
        if len(train_val_metadata) > 0:
            labels = train_val_metadata["label"].values
            indices = train_val_metadata.index.values
            
            # Stratified split
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_val_split, random_state=random_state)
            train_idx, val_idx = next(sss.split(indices, labels))
            
            if partition == "train":
                selected_indices = indices[train_idx]
            else:
                selected_indices = indices[val_idx]
        else:
            selected_indices = train_val_metadata.index.values
        
        dataset = VideoDataset(
            directory=directory,
            partition="train",  # Always use train directory since we deferred split
            preprocessor=preprocessor,
            indices=list(selected_indices)
        )
    else:
        raise ValueError(f"dataset_name={dataset_name}, should contain 'letters' or 'glosses'")

    loader = DataLoader(
        dataset = dataset,
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=shuffle
        )
        
    return loader



#def collate_fn(batch):
    ## (batch, t_max, 42)
    ##padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)

    ## (t_max, batch, 42)
    #padded_batch = pad_sequence(batch, padding_value=0)
    #lengths = torch.tensor([t.shape[0] for t in batch], dtype=torch.long)
    #return padded_batch, lengths
#loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
