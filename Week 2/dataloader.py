import json
import cv2
import numpy as np
from random import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import torch
from shutil import copy

from pathlib import Path
from typing import Union, Callable
import numpy.typing as npt

import pandas as pd
from collections import defaultdict, OrderedDict


# ====================================================================================
# this code silences output from MediaPipe that isn't important
# run prior to importing MediaPipe
import sys
import os
sys.unraisablehook = lambda unraisable: None
# Silence MediaPipe / GLOG
#os.environ["GLOG_minloglevel"] = "3"   # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
# Silence TensorFlow / TFLite
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=INFO, 1=WARN, 2=ERROR, 3=FATAL

#_devnull = os.open(os.devnull, os.O_WRONLY)
#_old_stdout = os.dup(1)
#_old_stderr = os.dup(2)

#os.dup2(_devnull, 1)
#os.dup2(_devnull, 2)
# ====================================================================================
# import MediaPipe stuff
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

def letter_to_label(letter: str) -> int:
    """converts ASL letter (A-Z, excluding J and Z) to integer label 0-24. """
    letter = letter.upper()
    if letter not in most_letters:
        raise ValueError(f"Invalid letter: {letter}")
    return most_letters.index(letter)

def label_to_letter(label: int) -> str:
    return most_letters[label]


class LettersDatasetProcessor:
    def __init__(self, src_directory="asl_letters", train_val_split=0.8, filter_to_landmarkable=False, included_letters: list[str] =None) -> None:
        self.train_val_split = train_val_split
        self.excluded_letters = ["del", "space", "J", "Z", "nothing"]

        if included_letters is not None:
            self.included_letters = included_letters
        else: 
            self.included_letters = list(most_letters)
        
        self.filter_to_landmarkable = filter_to_landmarkable
        self.landmark_checker = ImageToTensorPreprocessor(
            output_format="landmarks",
            draw_on_img=True,
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

        self.dataset = {"index": [], "partition": [], "label": [], "letter": []}

        self._process_dataset()
        print('processed dataset!')

        df = pd.DataFrame(self.dataset)
        df.to_csv(self.tgt_directory / "letters.csv", index=False)
    
    def _add_record(self, index: int, partition: str, label: int, letter: str):
        self.dataset["index"].append(index)
        self.dataset["partition"].append(partition)
        self.dataset["label"].append(label)
        self.dataset["letter"].append(letter)

    def _process_dataset(self):
        # add train and validation data: 3000 observations for each letter

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
            for example in letter.iterdir():
                if self.filter_to_landmarkable:
                    img = cv2.imread(Path.cwd() / example)
                    img_landmarks = self.landmark_checker(img)
                    if img_landmarks is None:
                        # Skip images where landmarks cannot be detected
                        continue
                
                random_number = random()
                if random_number > self.train_val_split:
                    copy(Path.cwd() / example, self.tgt_val / f"{self.val_index}.jpg")
                    self._add_record(self.val_index, "val", letter_to_label(letter.name), letter.name)
                    if self.filter_to_landmarkable:
                        cv2.imwrite(self.preview_val / f"val_{self.val_index}.jpg", img)
                    self.val_index += 1
                else:
                    copy(Path.cwd() / example, self.tgt_train / f"{self.train_index}.jpg")
                    self._add_record(self.train_index, "train", letter_to_label(letter.name), letter.name)
                    if self.filter_to_landmarkable:
                        cv2.imwrite(self.preview_train / f"train_{self.train_index}.jpg", img)
                    self.train_index += 1
            
        # add test data: one observation each
        for letter in self.included_letters:
            copy(self.src_test_dir / f"{letter}_test.jpg", self.tgt_test / f"{self.test_index}.jpg")
            self._add_record(self.test_index, "test", letter_to_label(letter), letter)
            self.test_index += 1


class WLASLDatasetProcessor:
    def __init__(self, src_directory="asl_glosses", train_val_split = 0.8, top_n: int = 5) -> None:
        self.train_val_split = train_val_split
        self.excluded_glosses = []

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

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

        for k in self.excluded_glosses:
            try:
                glosses_grouped.pop(k, None)
            except KeyError:
                pass

        self.top_n = len(glosses_grouped) if top_n is None else 5

        self.glosses = dict(
            sorted(glosses_grouped.items(), key=lambda item: len(item[1]), reverse=True)[:self.top_n]
            )

        self.glosses = OrderedDict(self.glosses.items())

        for k,v in list(self.glosses.items()):
            print(f"n occurances of {k}: {len(v)}")

        self.gloss_to_label = lambda gloss: list(self.glosses.keys()).index(gloss)

        self.tgt_directory = self.src_directory.parent / f"{self.src_directory.name}_processed"

        self.tgt_train = self.tgt_directory / "train"
        self.tgt_train.mkdir(parents=True, exist_ok=True)

        self.tgt_val = self.tgt_directory / "val"
        self.tgt_val.mkdir(parents=True, exist_ok=True)

        self.tgt_test = self.tgt_directory / "test"
        self.tgt_test.mkdir(parents=True, exist_ok=True)

        self.dataset = {"index": [], "partition": [], "label": [], "gloss": []}

        self._process_dataset()

        df = pd.DataFrame(self.dataset)
        df.to_csv(self.tgt_directory / "glosses.csv", index=False)
    
    def _add_record(self, index: int, partition: str, label: int, gloss: str):
        self.dataset["index"].append(index)
        self.dataset["partition"].append(partition)
        self.dataset["label"].append(label)
        self.dataset["gloss"].append(gloss)

    def _process_dataset(self):
        for gloss, examples in self.glosses.items():
            if gloss in self.excluded_glosses:
                continue

            for example in examples:
                random_number = random()
                if random_number > self.train_val_split:
                    copy(Path.cwd() / example["filepath"], self.tgt_val / f"{self.val_index}.mp4")
                    self._add_record(self.val_index, "val", self.gloss_to_label(gloss), gloss)
                    self.val_index += 1
                else:
                    copy(Path.cwd() / example["filepath"], self.tgt_train / f"{self.train_index}.mp4")
                    self._add_record(self.train_index, "train", self.gloss_to_label(gloss), gloss)
                    self.train_index += 1

def landmarks_to_tensor(right_hand_coords: list[(int, float, float)]) -> torch.Tensor:
    """assumes only right hand"""
    output_tensor = torch.zeros(42, dtype=torch.float32)
    coords = []
    for i,x,y in right_hand_coords:
        coords.extend([x,y])
    output_tensor = torch.tensor(coords, dtype=torch.float32)
    return output_tensor
                
class ImageToTensorPreprocessor():
    """
    landmark preprocessor: takes the original tensor and an optional normalizing tensor, and returns a transformed landmark tensor

    returns tensor representation of input image

    if draw_on_img==True, modifies img directly
    """
    def __init__(self,
                output_format: "image|landmarks" = "image",
                image_preprocessor: Callable = None,
                landmark_normalization_method: "per-frame-wrist|first-frame-wrist|None" = "per-frame-wrist",
                static_image_mode=True,
                draw_on_img=False,
                return_img=False,
                #max_hands=1, 
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

        self.max_hands = 1

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
            
    def landmark_preprocessor(self, landmark_tensor, normalization_tensor): 
        return landmark_tensor - normalization_tensor.repeat(21)

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
            self.timestamp_ms += 33  # ~30 fps
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

    def __call__(self, image: npt.ArrayLike, first_frame_landmark_tensor: torch.Tensor = None) -> torch.Tensor:
        #landmarks = self.detector.find_position(image)
        if image is None:
            raise ValueError("Input image is None. Cannot process.")
            
        #image = cv2.flip(image, 1)
        if self.output_format == "image":
            tensor = self.image_preprocessor(image)
            return tensor
        else:
            # image = self.image_preprocessor  
            hand_landmarks = self.image_to_hand_landmarks(image)
            if len(hand_landmarks) == 0:
                # Return None to signal that no hand was detected
                if self.return_img:
                    return None, image
                else:
                    return None

            right_hand_coords = hand_landmarks[0][1]
            landmark_tensor = landmarks_to_tensor(right_hand_coords=right_hand_coords)

            if self.landmark_normalization_method is None:
                pass
            elif self.landmark_normalization_method == "per-frame-wrist":
                landmark_tensor = self.landmark_preprocessor(landmark_tensor, normalization_tensor = landmark_tensor[0:2])
            elif self.landmark_normalization_method == "first-frame-wrist" and first_frame_landmark_tensor is not None:
                landmark_tensor = self.landmark_preprocessor(landmark_tensor, normalization_tensor = first_frame_landmark_tensor[0:2])

            if self.draw_on_img:
                image = self.draw_hand_landmarks(image)

            if self.return_img:
                return landmark_tensor, image
            else:
                return landmark_tensor

class ImageDataset(Dataset):
    def __init__(
            self,
            directory: Union[str, Path],
            partition: str = "train",
            preprocessor: Callable[[npt.ArrayLike], torch.Tensor] = None,
        ):
            self.partition = partition
            if partition not in ("train", "test", "val"):
                raise ValueError(f"Invalid partition specified - {partition}")
            self.directory = Path(directory)
            self.img_directory = self.directory / self.partition
            metadata = pd.read_csv(self.directory / "letters.csv")
            self.metadata = metadata[metadata["partition"] == self.partition]
            self.metadata.reset_index(drop=True, inplace=True)

            if preprocessor is not None:
                self.preprocessor = preprocessor
            else:
                self.preprocessor = ImageToTensorPreprocessor()


    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        label = self.metadata.iloc[index]["label"]
        letter = self.metadata.iloc[index]["letter"]
        image_path = str(self.img_directory / f"{index}.jpg")
        
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

class VideoDataset(Dataset):
    def __init__(
            self,
            directory: Union[str, Path],
            partition: str = "train",
            preprocessor: Callable[[npt.ArrayLike], torch.Tensor] = None,
        ):
            print(f"initializing {partition} dataloader...")
            self.partition = partition
            if partition not in ("train", "test", "val"):
                raise ValueError(f"Invalid partition specified - {partition}")
            self.directory = Path(directory)
            self.img_directory = self.directory / self.partition
            metadata = pd.read_csv(self.directory / "glosses.csv")
            self.metadata = metadata[metadata["partition"] == self.partition]
            self.metadata.reset_index(drop=True, inplace=True)

            if preprocessor is not None:
                self.preprocessor = preprocessor
            else:
                self.preprocessor = ImageToTensorPreprocessor(
                    output_format="landmarks",
                    landmark_normalization_method="per-frame-wrist",
                    static_image_mode=False, 
                    draw_on_img=False,
                    max_hands=1)
                
            self.landmark_tensors, self.labels = [], []
            self.__save_videos_to_memory()
            print(f"finished initializing {partition} dataloader!")

    def __save_videos_to_memory(self):
        for index, row in self.metadata.iterrows():
            label = row["label"]
            self.landmark_tensors.append(self.__load_video(index))
            self.labels.append(label)

    def __load_video(self, index) -> tuple[torch.Tensor, int]:
        video_reader = cv2.VideoCapture(self.img_directory / f"{index}.mp4")
        is_first_frame = True
        # used when normalizing landmarks via "first-frame-wrist"
        first_frame_landmark_tensor = torch.zeros(42, dtype=torch.float32)
        video = []
        while True:
            ret, frame = (video_reader.read())
            if not ret:
                break
            current_frame = self.preprocessor(frame, first_frame_landmark_tensor=first_frame_landmark_tensor)
            if current_frame is not None:
                video.append(current_frame)
                if is_first_frame:
                    first_frame_landmark_tensor = current_frame.detach().clone()
                    is_first_frame = False

        return torch.stack(video)
    
    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        #label = self.metadata.iloc[index]["label"]
        #return self.__load_video(index), label
        return self.landmark_tensors[index], self.labels[index]





def get_dataloader(dataset_name: str, partition: str, as_landmarks: bool = False, batch_size: int = 1, num_workers = 0, shuffle: bool = None) -> DataLoader:
    """
    dataset: "asl_letters" | "asl_letters_small" | "asl_glosses"
    partition: "train" | "val"

    as_landmarks: if true, load data as landmark tensors, otherwise load as image tensors
    """

    assert (partition == "train") or (partition == "val")
    dataset = None

    preprocessor = ImageToTensorPreprocessor(
        output_format="landmarks" if as_landmarks else "image",
        image_preprocessor=None,
        #landmark_normalization_method=None
        landmark_normalization_method="per-frame-wrist"
        )

    if dataset_name == "asl_letters" or dataset_name == "asl_letters_small":
        dataset = ImageDataset(
            directory=f"{dataset_name}_processed",
            partition=partition, 
            preprocessor=preprocessor)

    elif dataset_name == "asl_glosses":
        dataset = VideoDataset(
            directory="asl_glosses_processed",
            partition=partition, 
            preprocessor=preprocessor)
    else:
        raise ValueError(f"dataset_name={dataset_name}, should be one of 'asl_letters', 'asl_letters_small', 'asl_glosses")

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
