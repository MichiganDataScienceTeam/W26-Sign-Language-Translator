import shutil
from pathlib import Path
from dataloader import LettersDatasetProcessor

def process_dataset(dataset_name: str, included_letters=None, to_video: bool = False, max_examples_per_letter: int = 1000):
    processed_dir = Path(dataset_name).parent / f"{dataset_name}_processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    
    print(f"Processing {dataset_name} with landmark filtering (max {max_examples_per_letter} per letter)...")
    processor = LettersDatasetProcessor(
        src_directory=dataset_name,
        filter_to_landmarkable=True,
        included_letters=included_letters,
        to_video=to_video,
        max_examples_per_letter=max_examples_per_letter,
    )
    print(f"Dataset saved to {processed_dir}")




if __name__ == "__main__":
    # process_dataset("asl_letters_small", included_letters=["A", "B"], to_video=True)
    process_dataset("asl_letters", to_video=True, max_examples_per_letter = 1000)
