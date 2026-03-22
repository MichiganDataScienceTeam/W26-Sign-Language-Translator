#!/usr/bin/env python

from custom_dataset import CustomDatasetCreator, CustomDatasetProcessor

if __name__ == "__main__":
    data_creator = CustomDatasetCreator(dataset_name="hand_signs", data_type="video")
    data_creator(gesture_name = "dog")
    data_creator(gesture_name = "bird")
    processor = CustomDatasetProcessor(dataset_name = "hand_signs", data_type="video")
