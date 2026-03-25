"""
Training script for CBAM-CNN radar point quality classifier.

Generates multiple simulated scenarios, extracts 8-channel tensor patches,
and trains the CBAM-UNet model with focal loss.
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score

from src.data_simulation import RadarSimulator
from src.layering import Layerer
from src.model import CBAMUNet, FocalLoss, RadarDataset


def generate_training_data(config: dict, n_scenarios: int = 20,
                            patch_size: int = 64, stride: int = 32):
    """
    Generate training dataset from multiple simulated radar scenarios.

    Returns (patches, labels) numpy arrays.
    """
    all_patches = []
    all_labels = []

    print(f"Generating {n_scenarios} simulated scenarios...")
    for i in tqdm(range(n_scenarios)):
        # Vary random seed per scenario
        config_copy = dict(config)
        config_copy['simulation'] = dict(config['simulation'])
        config_copy['simulation']['seed'] = i * 7 + 42
        config_copy['simulation']['n_targets'] = np.random.randint(3, 10)
        config_copy['simulation']['clutter_lambda'] = np.random.randint(400, 1500)

        sim = RadarSimulator(config_copy)
        points = sim.simulate()

        from src.data_simulation import points_to_array
        pts_arr = points_to_array(points)

        layerer = Layerer(config)
        n_frames = int(pts_arr['frame_id'].max()) + 1

        for start in range(0, n_frames - 2, 3):
            fr, fg, fb = start, start + 1, start + 2
            tensor = layerer.build_tensor(pts_arr, fr, fg, fb)
            label_map = layerer.build_label_map(pts_arr, fr, fg, fb)
            patches, labels = layerer.extract_patches(
                tensor, label_map, patch_size=patch_size, stride=stride)
            if len(patches) > 0:
                all_patches.append(patches)
                all_labels.append(labels)

    if not all_patches:
        raise ValueError("No training patches generated!")

    patches = np.concatenate(all_patches, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"Total patches: {len(patches)}, "
          f"target pixels: {(labels == 1).sum()}, "
          f"clutter pixels: {(labels == 0).sum()}")
    return patches, labels


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for patches, labels in loader:
        patches = patches.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []
    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device)
            labels = labels.to(device)
            logits = model(patches)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]  # target confidence
            preds = (probs > 0.5).long()

            all_preds.append(preds.cpu().numpy().ravel())
            all_labels.append(labels.cpu().numpy().ravel())
            all_scores.append(probs.cpu().numpy().ravel())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)

    # Compute AUROC
    try:
        auroc = roc_auc_score(all_labels, all_scores)
    except Exception:
        auroc = 0.5

    avg_loss = total_loss / len(loader)
    return avg_loss, auroc, all_preds, all_labels, all_scores


def main():
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    model_cfg = config['model']
    patch_size = model_cfg['patch_size']
    batch_size = model_cfg['batch_size']
    n_epochs = model_cfg['n_epochs']
    lr = model_cfg['learning_rate']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate training data
    n_scenarios = 15  # Reduced for faster training; increase for publication
    patches, labels = generate_training_data(config, n_scenarios=n_scenarios,
                                              patch_size=patch_size, stride=32)

    # Train/val/test split
    n = len(patches)
    idx = np.random.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_patches = patches[idx[:n_train]]
    train_labels = labels[idx[:n_train]]
    val_patches = patches[idx[n_train:n_train + n_val]]
    val_labels = labels[idx[n_train:n_train + n_val]]
    test_patches = patches[idx[n_train + n_val:]]
    test_labels = labels[idx[n_train + n_val:]]

    train_ds = RadarDataset(train_patches, train_labels, augment=True)
    val_ds = RadarDataset(val_patches, val_labels, augment=False)
    test_ds = RadarDataset(test_patches, test_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = CBAMUNet(in_channels=model_cfg['n_channels'], n_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = FocalLoss(alpha=model_cfg['focal_alpha'], gamma=model_cfg['focal_gamma'])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,
                                                      factor=0.5, min_lr=1e-6)

    # Training loop
    os.makedirs('checkpoints', exist_ok=True)
    best_auroc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_auroc': []}

    print(f"\nTraining CBAM-UNet for {n_epochs} epochs...")
    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auroc, _, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auroc'].append(val_auroc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUROC: {val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')

    # Load best and evaluate on test set
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    test_loss, test_auroc, preds, labels_test, scores = eval_epoch(
        model, test_loader, criterion, device)

    print(f"\n=== Test Results ===")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(classification_report(labels_test, preds,
                                  target_names=['Clutter', 'Target']))

    # Plot training history
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs('outputs', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['val_auroc'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUROC')
    axes[1].set_title('Validation AUROC')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save classification results
    from src.visualization import plot_classification_metrics
    plot_classification_metrics(labels_test, preds, scores,
                                 'outputs/fig_H_classification_metrics.png')
    print("\nTraining complete. Best model: checkpoints/best_model.pth")
    print(f"Best Val AUROC: {best_auroc:.4f}, Test AUROC: {test_auroc:.4f}")


if __name__ == '__main__':
    main()
