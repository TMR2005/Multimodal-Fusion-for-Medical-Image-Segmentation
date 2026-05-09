import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
import os

def plot_training_curves(history, output_dir):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Val', marker='s')
    axes[0].set_title('Loss vs Epoch', fontweight='bold'); axes[0].legend(); axes[0].grid(alpha=0.3)
    
    # Dice
    axes[1].plot(history['train_dice'], label='Train', marker='o')
    axes[1].plot(history['val_dice'], label='Val', marker='s')
    axes[1].set_title('Dice Score vs Epoch', fontweight='bold'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # IoU
    axes[2].plot(history['val_iou'], marker='o', color='green')
    axes[2].set_title('IoU vs Epoch', fontweight='bold'); axes[2].grid(alpha=0.3); axes[2].set_ylim([0, 1])

    # Precision/Recall
    axes[3].plot(history.get('val_precision', []), label='Precision', marker='^')
    axes[3].plot(history.get('val_recall', []), label='Recall', marker='v')
    axes[3].set_title('Precision & Recall', fontweight='bold'); axes[3].legend(); axes[3].grid(alpha=0.3)
    axes[3].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
 
def visualize_segmentation(image, mask, prediction, patient_id, output_dir, prefix="02_segmentation"):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image, cmap='gray'); axes[0].set_title('Input FLAIR MRI (Center)'); axes[0].axis('off')
    axes[1].imshow(mask, cmap='tab10'); axes[1].set_title('Ground Truth Mask'); axes[1].axis('off')
    axes[2].imshow(prediction, cmap='tab10'); axes[2].set_title('Predicted Mask'); axes[2].axis('off')
    
    overlay = np.stack([image, image, image], axis=-1)
    overlay[prediction > 0] = [1, 0, 0]  # Red for predictions
    axes[3].imshow(overlay); axes[3].set_title('Prediction Overlay'); axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_{patient_id}.png", dpi=150, bbox_inches='tight')
    plt.close()
 
def plot_embedding_alignment(image_emb, text_emb, output_dir):
    all_emb = np.vstack([image_emb, text_emb])
    labels = ['Image'] * len(image_emb) + ['Text'] * len(text_emb)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_emb)-1))
    reduced = tsne.fit_transform(all_emb)
    
    plt.figure(figsize=(8, 6))
    for label in set(labels):
        mask = np.array(labels) == label
        plt.scatter(reduced[mask, 0], reduced[mask, 1], label=label, s=100, alpha=0.7)
    
    plt.title('Image-Text Embedding Alignment (t-SNE)', fontweight='bold')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{output_dir}/03_embedding_alignment.png", dpi=150)
    plt.close()

def plot_attention_map(image, features, patient_id, output_dir):
    """Rubric 6.3: Explainability via Bottleneck Feature Maps"""
    # Average features across channel dimension
    attention = torch.mean(features, dim=0).unsqueeze(0).unsqueeze(0) # [1, 1, 32, 32]
    # Upsample to image size
    attention = F.interpolate(attention, size=(256, 256), mode='bilinear', align_corners=False)
    attention = attention.squeeze().cpu().numpy()
    
    # Normalize 0-1
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray'); plt.title("Original MRI"); plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.imshow(attention, cmap='jet', alpha=0.5) # Overlay attention
    plt.title("Model Bottleneck Attention")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_attention_{patient_id}.png", dpi=150)
    plt.close()

def plot_error_distribution(dice_scores, output_dir):
    """Rubric 6.6: Error Analysis Distribution"""
    plt.figure(figsize=(8, 5))
    plt.hist(dice_scores, bins=20, color='coral', edgecolor='black')
    plt.axvline(np.mean(dice_scores), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(dice_scores):.3f}')
    plt.title('Validation Dice Score Distribution (Error Analysis)', fontweight='bold')
    plt.xlabel('Dice Score'); plt.ylabel('Frequency')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{output_dir}/05_error_distribution.png", dpi=150)
    plt.close()