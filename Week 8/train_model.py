"""
Train LSTM on ASL Citizen

Run processor first:
    python3 asl_citizen_processor.py --top-n 20

Then train:
    python3 train_asl_citizen.py --epochs 50 --hidden-size 256 --layers 4
"""

import torch
import pandas as pd
from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from lstm_model import Video_LSTM
from training   import train_model, visualize_results


# Dataset

class ASLDataset(Dataset):
    def __init__(self, directory="asl_citizen_processed", partition="train"):
        self.data_dir = Path(directory) / partition
        metadata      = pd.read_csv(Path(directory) / "glosses.csv")
        self.metadata = metadata[metadata["partition"] == partition].reset_index(drop=True)
        print(f"  {partition}: {len(self.metadata)} videos")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        video = torch.load(self.data_dir / f"{index}.pt")
        label = int(self.metadata.iloc[index]["label"])
        return video, label


def get_dataloaders(processed_dir="asl_citizen_processed"):
    train_ds     = ASLDataset(processed_dir, "train")
    val_ds       = ASLDataset(processed_dir, "val")
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
    return train_loader, val_loader


def get_class_weights(processed_dir, num_classes, device):
    """
    Compute inverse-frequency class weights so rare classes get
    higher loss penalty — prevents the model collapsing to majority classes.
    """
    metadata   = pd.read_csv(Path(processed_dir) / "glosses.csv")
    train_meta = metadata[metadata["partition"] == "train"]
    counts     = Counter(train_meta["label"].tolist())
    weights    = torch.tensor(
        [1.0 / (counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32
    )
    weights = weights / weights.sum() * num_classes  # normalize
    return weights.to(device)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(
    processed_dir = "asl_citizen_processed",
    epochs        = 50,
    lr            = 1e-4,
    n_layers      = 4,
    hidden_size   = 256,
    dropout       = 0.5,
    model_name    = "asl_citizen",
):
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        print(f"Processed data not found at '{processed_dir}'")
        return


    # Read config saved by processor
    cfg         = pd.read_csv(processed_path / "config.csv").iloc[0]
    feature_dim = int(cfg["feature_dim"])
    num_classes = int(cfg["num_classes"])

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("TRAINING ON ASL CITIZEN")
    print("=" * 70)
    print(f"  Feature dim  : {feature_dim}")
    print(f"  Classes      : {num_classes}")
    print(f"  Epochs       : {epochs}")
    print(f"  LR           : {lr}")
    print(f"  Hidden size  : {hidden_size}")
    print(f"  Layers       : {n_layers}")
    print(f"  Dropout      : {dropout}")
    print(f"  Device       : {device}")
    print()

    train_loader, val_loader = get_dataloaders(processed_dir)

    model = Video_LSTM(
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers=n_layers,
        num_classes=num_classes,
        input_size=feature_dim,
    )

    # Class-weighted loss 
    class_weights = get_class_weights(processed_dir, num_classes, device)
    criterion     = torch.nn.CrossEntropyLoss(weight=class_weights)

    # AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    results = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        save_prefix=model_name,
    )

    visualize_results(results, save_prefix=model_name)

    print()
    print("Training complete!")
    print(f"   Model  -> saved_models/{model_name}_fc_model.pth")
    print(f"   Plots  -> saved_plots/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", default="asl_citizen_processed")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--hidden-size",   type=int,   default=256)
    p.add_argument("--layers",        type=int,   default=4)
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--model-name",    default="asl_citizen")
    args = p.parse_args()

    train_model(
        processed_dir=args.processed_dir,
        epochs=args.epochs,
        lr=args.lr,
        n_layers=args.layers,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        model_name=args.model_name,
    )
