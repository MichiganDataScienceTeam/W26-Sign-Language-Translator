import torchvision.transforms.v2 as v2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using MPS")
    device = torch.device("mps")
else:
    print("Using CPU")
    device = torch.device("cpu")

kwargs = {'num_workers': 0}
if torch.cuda.is_available():
    kwargs['pin_memory'] = True

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, save_prefix: str = None):

    model.to(device)

    models_dir = "saved_models"
    os.makedirs(models_dir, exist_ok=True)

    #num_epochs = 30  # number of epochs to train the model
    best_val_acc = 0
    train_losses, val_losses = [], []  # lists to store the loss values over time
    train_accs, val_accs = [], []  # list to store the validation accuracy over time
    val_loss, val_acc = 0, 0

    # Training loop
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        train_count_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        # Training loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, (X, y) in enumerate(pbar):
            X = X.to(device)
            y = y.to(device)

            # Set the model to training mode
            model.train()

            # Clear the gradients
            optimizer.zero_grad()

            # Pass the inputs through the neural network
            logits = model(X)

            # Compute the loss
            loss = criterion(logits, y)
            running_train_loss += loss.item() * len(X)

            # Store predictions and labels for confusion matrix
            preds = logits.argmax(1)
            all_train_preds.extend(preds.cpu().detach().numpy())
            all_train_labels.extend(y.cpu().numpy())

            # Track training accuracy
            train_count_correct += torch.count_nonzero(logits.argmax(1) == y).item()
            train_total += len(X)

            # Backpropagate the gradients
            loss.backward()

            # Update the weights
            optimizer.step()

            loss_value = float(loss.item())
            pbar.set_postfix(
                loss=loss_value, val_loss=val_loss, val_acc=val_acc, refresh=False
            )

        # Calculate epoch-level training metrics
        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = train_count_correct / train_total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation loop
        val_loss = 0
        count_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            # Set the model to evaluation mode
            model.eval()

            # Pass the inputs through the neural network
            logits = model(X)

            # Compute the loss
            loss = criterion(logits, y)

            # Accumulate the loss
            val_loss += len(X) * float(loss.item())

            # Store predictions and labels for confusion matrix
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Accumulate the counter for correct predictions
            count_correct += torch.count_nonzero(
                logits.argmax(1) == y
            ).item()
            
            val_total += len(X)


        val_loss = val_loss / val_total
        val_losses.append(val_loss)

        val_acc = float(count_correct / val_total)
        val_accs.append(val_acc)

        pbar.set_postfix(
            loss=float(loss.item()), val_loss=val_loss, val_acc=val_acc
            )
        #if val_loss < min_val_loss:
            ## Update the minimum validation loss so far
            #min_val_loss = val_loss


    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(models_dir, f"{save_prefix}_fc_model.pth" if save_prefix else "fc_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"saved best model to {save_path}: {val_acc:2f}% accuracy")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "train_predictions": all_train_preds,
        "train_labels": all_train_labels,
        "val_predictions": all_preds,
        "val_labels": all_labels,
    }


def visualize_results(results: dict, save_prefix: str = ""):
    prefix = f"{save_prefix}_" if save_prefix else ""
    plots_dir = "saved_plots"
    os.makedirs(plots_dir, exist_ok=True)
    accuracy_plot = os.path.join(plots_dir, f"{prefix}accuracy.png")
    loss_plot = os.path.join(plots_dir, f"{prefix}losses.png")

    # Plot the training and validation loss curves
    plt.figure()
    epochs = np.arange(len(results["train_losses"]))
    plt.plot(epochs, results["train_losses"], color="C0", label="Training")
    plt.plot(epochs, results["val_losses"], label="Validation", color="C1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss: {save_prefix}")
    plt.legend()
    plt.savefig(loss_plot)
    #plt.show()

    # Plot the training and validation accuracy curves
    plt.figure()
    plt.plot(epochs, 100 * np.array(results["train_accs"]), color="C0", label="Training")
    plt.plot(epochs, 100 * np.array(results["val_accs"]), color="C1", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy: {save_prefix}")
    plt.legend()
    plt.savefig(accuracy_plot)

    print(f"losses saved to {loss_plot}")
    print(f"accuracies saved to {accuracy_plot}")

    # plot confusion matrix
    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################
    label_map_df = pd.read_csv("asl_citizen_processed/label_map.csv")
    class_names = label_map_df['gloss'].unique()
    plot_confusion_matrix(results, class_names, save_prefix)
    ####################################################################
    ####################################################################
    ####################################################################
    #plt.show()


def plot_confusion_matrix(results: dict, class_names=None, save_prefix: str = ""):
    prefix = f"{save_prefix}_" if save_prefix else ""
    #norm_suffix = "_normalized" if normalize else ""
    norm_suffix = ""

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    if class_names is not None:
        class_names = list(class_names)
        labels = list(range(len(class_names)))
    else:
        labels = None

    for idx, dataset in enumerate(["train", "val"]):
        y_true = results[f"{dataset}_labels"]
        y_pred = results[f"{dataset}_predictions"]

        cm = confusion_matrix(y_true, y_pred, labels=labels)
    
        # normalize by gloss frequency
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(
            cm.astype("float"),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
        fmt = '.2f'
        #fmt = 'd'

        annot_labels = np.where(cm > 0, np.char.mod('%.2f', cm), '')
        if len(class_names) > 45:
            annot_labels = False

        ax = axes[idx]
        sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', 
                    xticklabels=class_names if class_names is not None else labels,
                    yticklabels=class_names if class_names is not None else labels,
                    ax=ax)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        dataset_label = "Training" if dataset == "train" else "Validation"
        ax.set_title(f'{dataset_label} Confusion Matrix')
    
    fig.suptitle(f'Confusion Matrices: {save_prefix}', fontsize=16, y=1.02)
    
    plots_dir = "saved_plots"
    os.makedirs(plots_dir, exist_ok=True)
    confusion_plot = os.path.join(plots_dir, f"{prefix}confusion_matrix{norm_suffix}.png")
    plt.savefig(confusion_plot, bbox_inches='tight')
    print(f"confusion matrices saved to {confusion_plot}")
    plt.close()

