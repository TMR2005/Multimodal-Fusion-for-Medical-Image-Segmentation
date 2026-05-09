import os
import json
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

from PreProcessing import CustomBRATSDataset, DataStructureExplorer
from Utils import compute_dice, compute_iou, compute_precision_recall, compute_hausdorff
from Visualization import plot_training_curves, visualize_segmentation, plot_embedding_alignment, plot_attention_map, plot_error_distribution
from Models import MultimodalSegmentationModel, MultiModalLoss

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "results_dir": "./results",
    "image_size": 256,
    "batch_size": 2, # Keep small for 2.5D memory constraints
    "num_workers": 0,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "num_classes": 3,
    "embedding_dim": 512,
    "seg_weight": 1.0,
    "contrastive_weight": 0.1,
    "temperature": 0.07,
    "device": device,
    "random_seed": 42,
}

os.makedirs(CONFIG["results_dir"], exist_ok=True)
torch.manual_seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_dice = 0.0, 0.0
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images, masks, texts = batch["image"].to(device), batch["mask"].to(device), batch["text"]
        outputs = model(images, texts)
        loss_dict = criterion(outputs, masks)
        loss = loss_dict["total_loss"]
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            dice = compute_dice(outputs["segmentation"], masks)
            
        total_loss += loss.item()
        total_dice += dice.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice.item():.4f}"})
        
    return total_loss / len(loader), total_dice / len(loader)

def validate(model, loader, criterion, device, return_individual_scores=False, is_ablation=False):
    model.eval()
    metrics = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "hausdorff": 0.0}
    individual_scores = [] # For error analysis
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation (Ablation)" if is_ablation else "Validation"):
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            texts = [""] * len(images) if is_ablation else batch["text"] # Ablation: Empty Text
            
            outputs = model(images, texts)
            loss_dict = criterion(outputs, masks)
            
            dice = compute_dice(outputs['segmentation'], masks)
            iou = compute_iou(outputs['segmentation'], masks)
            precision, recall = compute_precision_recall(outputs['segmentation'], masks)
            hd95 = compute_hausdorff(outputs['segmentation'], masks)
            
            metrics["loss"] += loss_dict["total_loss"].item()
            metrics["dice"] += dice.item()
            metrics["iou"] += iou.item()
            metrics["precision"] += precision.item()
            metrics["recall"] += recall.item()
            metrics["hausdorff"] += hd95
            
            if return_individual_scores:
                for i in range(len(images)):
                    single_dice = compute_dice(outputs['segmentation'][i:i+1], masks[i:i+1]).item()
                    individual_scores.append({
                        "patient_id": batch["patient_id"][i],
                        "dice": single_dice,
                        "image": images[i].cpu(),
                        "mask": masks[i].cpu(),
                        "pred": torch.argmax(outputs['segmentation'][i:i+1], dim=1).squeeze(0).cpu()
                    })

    for k in metrics.keys(): metrics[k] /= len(loader)
    
    if return_individual_scores: return metrics, individual_scores
    return metrics

def main():
    print(f"Using device: {CONFIG['device']}")
    train_dataset = CustomBRATSDataset(split="train", image_size=CONFIG["image_size"])
    val_dataset = CustomBRATSDataset(split="val", image_size=CONFIG["image_size"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, pin_memory=True)

    model = MultimodalSegmentationModel(num_classes=CONFIG["num_classes"], embedding_dim=CONFIG["embedding_dim"]).to(CONFIG["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = MultiModalLoss(seg_weight=CONFIG["seg_weight"], contrastive_weight=CONFIG["contrastive_weight"], temperature=CONFIG["temperature"])

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [], "val_iou": [], "val_precision": [], "val_recall": []}
    best_val_dice = 0.0

    print("\nSTEP 1: Training")
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch [{epoch+1}/{CONFIG['num_epochs']}]")
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, criterion, CONFIG["device"])
        val_metrics = validate(model, val_loader, criterion, CONFIG["device"])

        history["train_loss"].append(train_loss); history["val_loss"].append(val_metrics["loss"])
        history["train_dice"].append(train_dice); history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])

        print(f"Val Dice: {val_metrics['dice']:.4f} | Val IoU: {val_metrics['iou']:.4f} | HD95: {val_metrics['hausdorff']:.2f}")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), os.path.join(CONFIG["results_dir"], "best_model.pt"))

    # =========================================================================
    # STEP 2 — ABLATION STUDY (Text vs No Text)
    # =========================================================================
    print("\nSTEP 2: Ablation Study (Evaluating Without Text)")
    model.load_state_dict(torch.load(os.path.join(CONFIG["results_dir"], "best_model.pt")))
    
    # Evaluate WITH text (Baseline)
    val_baseline, individual_scores = validate(model, val_loader, criterion, CONFIG["device"], return_individual_scores=True)
    
    # Evaluate WITHOUT text (Ablation)
    val_ablated = validate(model, val_loader, criterion, CONFIG["device"], is_ablation=True)
    
    print("\n--- Ablation Results ---")
    print(f"With Text (Baseline): Dice = {val_baseline['dice']:.4f}, IoU = {val_baseline['iou']:.4f}")
    print(f"Without Text (Ablated): Dice = {val_ablated['dice']:.4f}, IoU = {val_ablated['iou']:.4f}")

    # =========================================================================
    # STEP 3 — VISUALIZATIONS & ERROR ANALYSIS
    # =========================================================================
    print("\nSTEP 3: Visualizations & Error Analysis")
    plot_training_curves(history, CONFIG["results_dir"])
    
    # 6.6 Error Analysis - Plot Histogram
    dice_scores = [s['dice'] for s in individual_scores]
    plot_error_distribution(dice_scores, CONFIG["results_dir"])
    
    # 6.6 Error Analysis - Find and Plot 3 Worst Cases
    individual_scores.sort(key=lambda x: x['dice'])
    worst_cases_dir = os.path.join(CONFIG["results_dir"], "error_analysis_worst_cases")
    os.makedirs(worst_cases_dir, exist_ok=True)
    
    for i in range(3):
        sample = individual_scores[i]
        center_slice = sample['image'][1].numpy() # Index 1 is center slice for 2.5D
        visualize_segmentation(center_slice, sample['mask'].numpy(), sample['pred'].numpy(), 
                             sample['patient_id'], worst_cases_dir, prefix=f"worst_case_{i+1}")

    # 6.3 Explainability - Attention Maps
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        images, texts = batch["image"].to(CONFIG["device"]), batch["text"]
        outputs = model(images, texts)
        
        # Plot attention for first image
        center_slice = images[0, 1].cpu().numpy()
        features = outputs['features'][0] # Bottleneck features for first image
        plot_attention_map(center_slice, features, batch["patient_id"][0], CONFIG["results_dir"])
        
        # 6.5 Embedding Analysis
        image_emb = outputs["image_embeddings"].cpu().numpy()
        text_emb = outputs["text_embeddings"].cpu().numpy()
        plot_embedding_alignment(image_emb, text_emb, CONFIG["results_dir"])

    print("\nPIPELINE COMPLETE! Check results folder.")

if __name__ == "__main__":
    main()