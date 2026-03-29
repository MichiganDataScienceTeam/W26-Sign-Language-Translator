"""
Usage:
    python merge_datasets.py asl_glosses_processed asl_citizen_processed --output merged_processed
    python merge_datasets.py asl_glosses_processed --output filtered_processed --glosses DOG BASKETBALL
    python merge_datasets.py a_processed b_processed --output merged_processed --map-gloss dog=DOG,dog,Dog
"""

import shutil
from pathlib import Path
import pandas as pd


def _norm_gloss(gloss) -> str:
    """Canonical gloss key used to merge equivalent glosses across datasets."""
    return str(gloss)


def parse_gloss_mappings(mapping_args: list[str] | None) -> dict[str, str]:
    """
    Parse mapping entries in the form:
        DEST=SRC1,SRC2,SRC3

    Returns a dict mapping each source gloss to destination gloss.
    The destination gloss maps to itself as well.
    """
    alias_to_dest = {}
    if not mapping_args:
        return alias_to_dest

    for entry in mapping_args:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --map-gloss value '{entry}'. Expected DEST=SRC1,SRC2,..."
            )
        dest, sources = entry.split("=", 1)
        dest = dest.strip()
        if dest == "":
            raise ValueError(f"Invalid --map-gloss value '{entry}': destination is empty")

        source_items = [s.strip() for s in sources.split(",") if s.strip() != ""]
        if not source_items:
            raise ValueError(
                f"Invalid --map-gloss value '{entry}': must provide at least one source gloss"
            )

        for src in source_items + [dest]:
            if src in alias_to_dest and alias_to_dest[src] != dest:
                raise ValueError(
                    f"Conflicting mapping for gloss '{src}': "
                    f"'{alias_to_dest[src]}' vs '{dest}'"
                )
            alias_to_dest[src] = dest

    return alias_to_dest


def print_glosses(processed_dir: str):
    """
    Print all glosses in a processed dataset with frequencies and labels.
    
    Args:
        processed_dir: Path to _processed directory
    """
    processed_dir = Path(processed_dir)
    
    if not (processed_dir / "label_map.csv").exists():
        print(f"Error: {processed_dir} does not contain label_map.csv")
        return
    
    label_map = pd.read_csv(processed_dir / "label_map.csv")
    glosses_csv = pd.read_csv(processed_dir / "glosses.csv")
    
    gloss_counts = glosses_csv['gloss'].value_counts().to_dict()
    
    print(f"\nGlosses in {processed_dir.name}:")
    print("=" * 70)
    print(f"{'Label':<8} {'Gloss':<40} {'Frequency':<10}")
    print("-" * 70)
    
    for _, row in label_map.iterrows():
        label = int(row['label'])
        gloss = row['gloss']
        freq = gloss_counts.get(gloss, 0)
        print(f"{label:<8} {gloss:<40} {freq:<10}")
    
    print("=" * 70)
    print(f"Total glosses: {len(label_map)}")
    print(f"Total samples: {len(glosses_csv)}")


def merge_datasets(
    processed_dirs: list[str],
    output_dir: str = None,
    selected_glosses: list[str] = None,
    excluded_glosses: list[str] = None,
    gloss_aliases: dict[str, str] = None,
    overwrite: bool = False
):
    """
    Merge multiple processed datasets into a single dataset.
    
    Each unique gloss is assigned a globally unique label in the output.
    Glosses are sorted alphabetically for stability.
    
    Args:
        processed_dirs: List of paths to _processed directories to merge
        output_dir: Path to output merged _processed directory
        selected_glosses: Optional list of glosses to include (default: all)
        excluded_glosses: Optional list of glosses to exclude
        gloss_aliases: Optional mapping from source gloss -> destination gloss
        overwrite: Whether to overwrite existing output directory
    """
    if output_dir is None:
        raise ValueError("output_dir is required when merging datasets")

    if gloss_aliases is None:
        gloss_aliases = {}

    def canonical_gloss(gloss):
        g = _norm_gloss(gloss)
        return gloss_aliases.get(g, g)

    processed_dirs = [Path(d) for d in processed_dirs]
    output_dir = Path(output_dir)
    
    # Check if output exists
    if output_dir.exists() and not overwrite:
        print(f"Error: {output_dir} already exists. Use --overwrite to replace.")
        return
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "mp4_examples").mkdir(parents=True, exist_ok=True)
    
    print(f"Merging {len(processed_dirs)} dataset(s)...")
    
    # Collect all unique glosses from glosses.csv to capture actual sample labels.
    all_glosses_set = set()
    for pdir in processed_dirs:
        glosses_csv_path = pdir / "glosses.csv"
        if glosses_csv_path.exists():
            glosses_df = pd.read_csv(glosses_csv_path)
            for gloss in glosses_df["gloss"].tolist():
                mapped_gloss = canonical_gloss(gloss)
                if mapped_gloss == "":
                    continue
                all_glosses_set.add(mapped_gloss)
        else:
            print(f"Warning: {pdir} has no glosses.csv, skipping gloss discovery")
    
    # Filter glosses if specified
    if selected_glosses is not None:
        selected_glosses_set = {canonical_gloss(g) for g in selected_glosses}
        excluded = all_glosses_set - selected_glosses_set
        if excluded:
            print(f"Filtering out {len(excluded)} gloss(es): {', '.join(sorted(excluded)[:5])}...")
        all_glosses_set = {g for g in all_glosses_set if g in selected_glosses_set}

    # Exclude glosses if specified
    if excluded_glosses is not None:
        excluded_glosses_set = {canonical_gloss(g) for g in excluded_glosses}
        actually_excluded = all_glosses_set.intersection(excluded_glosses_set)
        if actually_excluded:
            print(
                f"Excluding {len(actually_excluded)} gloss(es): "
                f"{', '.join(sorted(actually_excluded)[:5])}" +
                ("..." if len(actually_excluded) > 5 else "")
            )
        all_glosses_set = {g for g in all_glosses_set if g not in excluded_glosses_set}
    
    # Create mapping from gloss to global label (alphabetically sorted for stability)
    all_glosses_sorted = sorted(all_glosses_set)
    gloss_to_global_label = {g: i for i, g in enumerate(all_glosses_sorted)}
    
    print(f"Total unique glosses: {len(all_glosses_sorted)}")
    print(f"Glosses: {', '.join(all_glosses_sorted[:10])}{'...' if len(all_glosses_sorted) > 10 else ''}")
    
    # Process each input dataset
    global_index = {"train": 0, "val": 0, "test": 0}
    merged_records = {
        "index": [],
        "partition": [],
        "label": [],
        "gloss": [],
        "original_filename": []
    }
    mp4_example_by_gloss = {}
    
    for pdir in processed_dirs:
        print(f"\nProcessing {pdir.name}...")
        
        glosses_csv_path = pdir / "glosses.csv"
        if not glosses_csv_path.exists():
            print(f"  Warning: no glosses.csv found, skipping")
            continue
        
        glosses_csv = pd.read_csv(glosses_csv_path)
        samples_processed = 0
        samples_filtered = 0
        
        # Process each sample
        for _, row in glosses_csv.iterrows():
            raw_gloss = row['gloss']
            mapped_gloss = canonical_gloss(raw_gloss)
            partition = row['partition']
            old_index = int(row['index'])
            
            # Skip if gloss not in selected glosses
            if mapped_gloss not in gloss_to_global_label:
                samples_filtered += 1
                continue
            
            global_label = gloss_to_global_label[mapped_gloss]
            output_gloss = mapped_gloss
            new_index = global_index[partition]
            
            # Copy .pt file
            src_pt = pdir / partition / f"{old_index}.pt"
            dst_pt = output_dir / partition / f"{new_index}.pt"
            if src_pt.exists():
                shutil.copy2(src_pt, dst_pt)
                samples_processed += 1
            else:
                print(f"    Warning: {src_pt} not found")
                continue
            
            # Record metadata
            merged_records["index"].append(new_index)
            merged_records["partition"].append(partition)
            merged_records["label"].append(global_label)
            merged_records["gloss"].append(output_gloss)
            merged_records["original_filename"].append(row.get('original_filename', ''))
            
            global_index[partition] += 1
        
        # Copy mp4 examples
        mp4_dir = pdir / "mp4_examples"
        if mp4_dir.exists():
            mp4_copied = 0
            for mp4_file in mp4_dir.glob("*.mp4"):
                mapped_gloss = canonical_gloss(mp4_file.stem)
                if mapped_gloss in gloss_to_global_label:
                    output_name = f"{mapped_gloss}.mp4"
                    dst_mp4 = output_dir / "mp4_examples" / output_name
                    if mapped_gloss not in mp4_example_by_gloss:
                        shutil.copy2(mp4_file, dst_mp4)
                        mp4_example_by_gloss[mapped_gloss] = output_name
                    mp4_copied += 1
        
        print(f" Processed {samples_processed} samples" + 
              (f" (filtered {samples_filtered})" if samples_filtered > 0 else ""))
    
    # Write output metadata
    print(f"\nWriting merged metadata...")
    
    merged_df = pd.DataFrame(merged_records)
    merged_df.to_csv(output_dir / "glosses.csv", index=False)
    print(f" glosses.csv ({len(merged_df)} rows)")
    
    # Create label_map
    label_map_data = []
    for mapped_gloss, label in gloss_to_global_label.items():
        mp4_example = mp4_example_by_gloss.get(mapped_gloss, "")
        label_map_data.append({
            "label": label,
            "gloss": mapped_gloss,
            "mp4_example": mp4_example
        })
    label_map_df = pd.DataFrame(label_map_data)
    label_map_df.to_csv(output_dir / "label_map.csv", index=False)
    print(f" label_map.csv ({len(label_map_df)} rows)")
    
    # Create config
    config_df = pd.DataFrame({
        "feature_dim": [84],
        "num_classes": [len(all_glosses_sorted)]
    })
    config_df.to_csv(output_dir / "config.csv", index=False)
    print(f" config.csv (feature_dim=84, num_classes={len(all_glosses_sorted)})")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Merged dataset created: {output_dir}")
    print(f"{'='*70}")
    print(f"Total samples: {len(merged_records['index'])}")
    print(f"  Train: {global_index['train']}")
    print(f"  Val:   {global_index['val']}")
    print(f"  Test:  {global_index['test']}")
    print(f"Total unique glosses: {len(all_glosses_sorted)}")
    print(f"Files with mp4 examples: {len(mp4_example_by_gloss)}/{len(all_glosses_sorted)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge multiple processed ASL datasets into a single dataset"
    )
    parser.add_argument(
        "input_dirs",
        nargs="*",
        help="Input _processed directories to merge"
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Output _processed directory path"
    )
    parser.add_argument(
        "--glosses",
        nargs="+",
        help="Optional: specific glosses to include (default: all)"
    )
    parser.add_argument(
        "--exclude-glosses",
        nargs="+",
        help="Optional: specific glosses to exclude"
    )
    parser.add_argument(
        "--map-gloss",
        action="append",
        help=(
            "Optional gloss remapping rule. Repeatable. "
            "Format: DEST=SRC1,SRC2,...  Example: dog=DOG,dog,Dog"
        ),
    )
    parser.add_argument(
        "--print-glosses",
        metavar="DIR",
        help="Print glosses in a _processed directory and exit"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory"
    )
    
    args = parser.parse_args()
    
    # Handle print-glosses mode
    if args.print_glosses:
        print_glosses(args.print_glosses)
    else:
        if not args.input_dirs:
            parser.error("input_dirs are required unless --print-glosses is used")
        if not args.output:
            parser.error("--output is required when merging datasets")

        try:
            gloss_aliases = parse_gloss_mappings(args.map_gloss)
        except ValueError as e:
            parser.error(str(e))

        # Merge mode
        merge_datasets(
            processed_dirs=args.input_dirs,
            output_dir=args.output,
            selected_glosses=args.glosses,
            excluded_glosses=args.exclude_glosses,
            gloss_aliases=gloss_aliases,
            overwrite=args.overwrite
        )
