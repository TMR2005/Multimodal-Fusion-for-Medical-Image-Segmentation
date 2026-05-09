import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import CLIPStyleTextEncoder

# =========================================================
# SEGMENTATION BACKBONE (U-Net)
# =========================================================

class SegmentationBackbone(nn.Module):
    """
    U-Net style encoder-decoder for BraTS segmentation

    Input:
        [B, in_channels, 256, 256]

    Output:
        seg_logits  -> [B, out_channels, 256, 256]
        bottleneck  -> [B, 256, 32, 32]
    """

    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()

        # ============================================================
        # ENCODER
        # ============================================================
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)

        # ============================================================
        # BOTTLENECK
        # ============================================================
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # ============================================================
        # DECODER
        # ============================================================
        self.upconv3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(pool3)

        # Decoder
        up3 = self.upconv3(bottleneck)
        up3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(up3)

        up2 = self.upconv2(dec3)
        up2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(up2)

        up1 = self.upconv1(dec2)
        up1 = torch.cat([up1, enc1], dim=1)
        
        seg_logits = self.dec1(up1)

        return seg_logits, bottleneck


# =========================================================
# MULTIMODAL MODEL
# =========================================================

class MultimodalSegmentationModel(nn.Module):

    def __init__(self, num_classes=4, embedding_dim=512):
        super().__init__()

        # Initialize U-Net expecting 3 channels for 2.5D data
        self.seg_backbone = SegmentationBackbone(
            in_channels=3,
            out_channels=num_classes
        )

        self.text_encoder = CLIPStyleTextEncoder(
            embedding_dim=embedding_dim
        )

        # Maps U-Net bottleneck (256 channels) to shared embedding space
        self.image_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512, embedding_dim)
        )

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

    def forward(self, images, texts):
        # 1. Image Forward Pass
        seg_logits, seg_features = self.seg_backbone(images)

        # 2. Text Forward Pass
        text_embeddings = self.text_encoder(texts)

        # 3. Project & Normalize Image Features
        image_embeddings = self.image_projection(seg_features)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

        # 4. Normalize Text Features
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        return {
            'segmentation': seg_logits,
            'image_embeddings': image_embeddings,
            'text_embeddings': text_embeddings,
            'features': seg_features
        }


# =========================================================
# MULTIMODAL LOSS
# =========================================================

class MultiModalLoss(nn.Module):

    def __init__(self, seg_weight=1.0, contrastive_weight=0.1, temperature=0.07):
        super().__init__()

        self.seg_weight = seg_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

        self.seg_loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, masks):
        
        # 1. Segmentation Loss
        seg_logits = outputs['segmentation']
        seg_loss = self.seg_loss_fn(seg_logits, masks.long())

        # 2. Contrastive Loss (CLIP-style)
        image_emb = outputs['image_embeddings']
        text_emb = outputs['text_embeddings']

        # Calculate cosine similarity logits scaled by temperature
        logits = (image_emb @ text_emb.T) / self.temperature

        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)

        # Bidirectional cross entropy (Image->Text and Text->Image)
        loss_img = F.cross_entropy(logits, labels)
        loss_txt = F.cross_entropy(logits.T, labels)
        contrastive_loss = (loss_img + loss_txt) / 2

        # 3. Total Combined Loss
        total_loss = (self.seg_weight * seg_loss) + (self.contrastive_weight * contrastive_loss)

        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss.item(),
            'contrastive_loss': contrastive_loss.item()
        }