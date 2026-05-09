from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# Suppress HuggingFace "UNEXPECTED" classification head warnings
logging.set_verbosity_error()

class CLIPStyleTextEncoder(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', embedding_dim=512, max_length=128, freeze_backbone=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.text_model.parameters():
                param.requires_grad = False
        hidden_size = self.text_model.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 768), nn.GELU(), nn.Dropout(0.1), nn.Linear(768, embedding_dim)
        )
        self.max_length = max_length

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    def forward(self, texts):
        device = next(self.projection.parameters()).device
        encoded = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = self.text_model(**encoded)
        text_features = self.mean_pooling(outputs, encoded['attention_mask'])
        embeddings = self.projection(text_features)
        return F.normalize(embeddings, p=2, dim=-1)

def compute_dice(preds, targets, num_classes=4, smooth=1e-5):
    preds = torch.argmax(preds, dim=1)
    dice_scores = []
    for cls in range(1, num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        if target_cls.sum() == 0: continue
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice_scores.append((2.0 * intersection + smooth) / (union + smooth))
    if len(dice_scores) == 0: return torch.tensor(1.0, device=preds.device)
    return torch.mean(torch.stack(dice_scores))

def compute_iou(preds, targets, num_classes=4, smooth=1e-5):
    preds = torch.argmax(preds, dim=1)
    iou_scores = []
    for cls in range(1, num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        if target_cls.sum() == 0: continue
        intersection = (pred_cls * target_cls).sum()
        union = (pred_cls.sum() + target_cls.sum() - intersection)
        iou_scores.append((intersection + smooth) / (union + smooth))
    if len(iou_scores) == 0: return torch.tensor(1.0, device=preds.device)
    return torch.mean(torch.stack(iou_scores))

def compute_precision_recall(preds, targets, num_classes=4, smooth=1e-5):
    preds = torch.argmax(preds, dim=1)
    precisions, recalls = [], []
    for cls in range(1, num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        if target_cls.sum() == 0: continue
        tp = (pred_cls * target_cls).sum()
        fp = (pred_cls * (1 - target_cls)).sum()
        fn = ((1 - pred_cls) * target_cls).sum()
        
        precisions.append((tp + smooth) / (tp + fp + smooth))
        recalls.append((tp + smooth) / (tp + fn + smooth))
        
    if len(precisions) == 0: 
        return torch.tensor(1.0, device=preds.device), torch.tensor(1.0, device=preds.device)
    return torch.mean(torch.stack(precisions)), torch.mean(torch.stack(recalls))

def compute_hausdorff(preds, targets, num_classes=4):
    """Computes approximation of Hausdorff Distance using scipy"""
    preds_np = torch.argmax(preds, dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    batch_hd = []
    for b in range(preds_np.shape[0]):
        # Just compute for the whole tumor area (any class > 0)
        pred_coords = np.argwhere(preds_np[b] > 0)
        target_coords = np.argwhere(targets_np[b] > 0)
        
        if len(pred_coords) == 0 or len(target_coords) == 0:
            batch_hd.append(100.0) # Penalty for completely missing
            continue
            
        hd1 = directed_hausdorff(pred_coords, target_coords)[0]
        hd2 = directed_hausdorff(target_coords, pred_coords)[0]
        batch_hd.append(max(hd1, hd2))
        
    return np.mean(batch_hd)