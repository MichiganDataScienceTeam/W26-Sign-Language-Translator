import shutil
from pathlib import Path
from dataloader import VideoDatasetProcessor

def process_dataset_exclude_glosses(dataset_name: str, top_n: int = 5, excluded_glosses: list[str] = None):
    processed_dir = Path(dataset_name).parent / f"{dataset_name}_processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    print(f"Processing {dataset_name} with landmark filtering...")
    processor = VideoDatasetProcessor(dataset_name, filter_to_landmarkable=True, top_n=top_n, excluded_glosses=excluded_glosses)
    print(f"Dataset saved to {processed_dir}")

def process_dataset_select_glosses(dataset_name: str, selected_glosses: list[str] = None):
    processed_dir = Path(dataset_name).parent / f"{dataset_name}_processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    print(f"Processing {dataset_name} with landmark filtering and gloss selection...")
    processor = VideoDatasetProcessor(dataset_name, filter_to_landmarkable=True, selected_glosses=selected_glosses)
    print(f"Dataset saved to {processed_dir}")

if __name__ == "__main__":
    #process_dataset("asl_glosses", top_n = 5)
    #process_dataset("asl_glosses", top_n = 2, excluded_glosses=["cool"])
    #process_dataset("asl_glosses", top_n = 2, excluded_glosses=["cool", "before", "thin", "drink", "go"])
    
    # current
    process_dataset_exclude_glosses("asl_glosses", top_n = 50, excluded_glosses=["cool", "before", "thin", "drink", "go"])

    # selected glosses
    #selected_glosses = ["able", "accept", "angry", "accent", "abdomen", "ago", "always", "cool", "before", "thin", "drink", "go", "computer"]
    #process_dataset_select_glosses("asl_glosses", selected_glosses=selected_glosses)
