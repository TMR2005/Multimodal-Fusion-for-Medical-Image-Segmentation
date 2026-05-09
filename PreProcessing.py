import os
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class DataStructureExplorer:
    @staticmethod
    def explore():
        print("\n" + "=" * 80)
        print("EXPLORING DATASET")
        print("=" * 80)
        img_path = Path("data/FLAIR_BRATS2020_split/train/images")
        mask_path = Path("data/FLAIR_BRATS2020_split/train/masks")
        text_path = Path("data/TextBRats/TextBraTSData")

        img_files = sorted(img_path.glob("image_*.npy"))
        print(f"\nFound {len(img_files)} MRI volumes")
        mask_files = sorted(mask_path.glob("mask_*.npy"))
        print(f"Found {len(mask_files)} mask volumes")
        patient_dirs = sorted([d for d in text_path.glob("BraTS20_*") if d.is_dir()])
        print(f"Found {len(patient_dirs)} patient text folders")

class PatientImageMapper:
    def __init__(self, flair_img_dir, flair_mask_dir, text_dir):
        self.flair_img_dir = Path(flair_img_dir)
        self.flair_mask_dir = Path(flair_mask_dir)
        self.text_dir = Path(text_dir)

        self.img_files = sorted(self.flair_img_dir.glob("image_*.npy"))
        self.mask_files = sorted(self.flair_mask_dir.glob("mask_*.npy"))
        self.patient_dirs = sorted([d for d in self.text_dir.glob("BraTS20_*") if d.is_dir()])
        self.mapping = self._build_mapping()

    def extract_image_id(self, filename):
        match = re.search(r'image_(\d+)', filename)
        return int(match.group(1)) if match else None

    def extract_patient_id(self, dirname):
        match = re.search(r'Training_(\d+)', dirname)
        return int(match.group(1)) if match else None

    def _find_text_file(self, patient_dir):
        txt_files = list(patient_dir.glob("*.txt"))
        return txt_files[0] if len(txt_files) > 0 else None

    def _build_mapping(self):
        mapping = {}
        patient_dict = {self.extract_patient_id(d.name): d for d in self.patient_dirs if self.extract_patient_id(d.name) is not None}

        for img_file, mask_file in zip(self.img_files, self.mask_files):
            img_id = self.extract_image_id(img_file.name)
            if img_id is None: continue
            
            patient_id = img_id + 1
            if patient_id not in patient_dict: continue
            
            patient_dir = patient_dict[patient_id]
            mapping[img_id] = {
                "img_file": img_file, "mask_file": mask_file,
                "patient_dir": patient_dir, "patient_id": patient_dir.name,
                "text_file": self._find_text_file(patient_dir)
            }
        return mapping

    def get_sample(self, idx):
        if idx not in self.mapping: return None
        info = self.mapping[idx]
        image = np.load(info["img_file"]).astype(np.float32)
        mask = np.load(info["mask_file"])
        text = "No report available."
        
        if info["text_file"] is not None:
            try:
                with open(info["text_file"], "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
            except Exception: pass
            
        return {
            "image": image, "mask": mask, "text": text,
            "patient_id": info["patient_id"], "idx": idx
        }

class CustomBRATSDataset(Dataset):
    def __init__(self, split="train", image_size=256, normalize=True):
        self.split = split
        self.image_size = image_size
        self.normalize = normalize

        img_dir = f"data/FLAIR_BRATS2020_split/{split}/images"
        mask_dir = f"data/FLAIR_BRATS2020_split/{split}/masks"

        self.mapper = PatientImageMapper(img_dir, mask_dir, "data/TextBRats/TextBraTSData")
        self.valid_indices = self._filter_valid_samples()
        print(f"\n{split.upper()} VALID SAMPLES (2.5D): {len(self.valid_indices)}")

    def _filter_valid_samples(self):
        valid = []
        for idx in self.mapper.mapping.keys():
            sample = self.mapper.get_sample(idx)
            if sample is None: continue
            mask = sample["mask"]
            middle_mask = np.argmax(mask[:, :, mask.shape[2] // 2, :], axis=-1) if mask.ndim == 4 else mask[:, :, mask.shape[2] // 2]
            if np.sum(middle_mask > 0) > 50:
                valid.append(idx)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def _normalize_image(self, image):
        img_min, img_max = image.min(), image.max()
        return (image - img_min) / (img_max - img_min) if img_max > img_min else np.zeros_like(image).astype(np.float32)

    def _resize_2d(self, slice_2d, is_mask=False):
        pil_img = Image.fromarray((slice_2d * (1 if is_mask else 255)).astype(np.uint8))
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        resized = pil_img.resize((self.image_size, self.image_size), resample)
        return np.array(resized).astype(np.int64) if is_mask else np.array(resized).astype(np.float32) / 255.0

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sample = self.mapper.get_sample(real_idx)

        if sample is None:
            return {
                "image": torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32),
                "mask": torch.zeros((self.image_size, self.image_size), dtype=torch.long),
                "text": "No report available.", "patient_id": "unknown", "idx": real_idx
            }

        image, mask = sample["image"], sample["mask"]
        mask_volume = np.argmax(mask, axis=-1) if mask.ndim == 4 else mask
        
        slice_sums = [np.sum(mask_volume[:, :, i] > 0) for i in range(mask_volume.shape[2])]
        best_z = np.argmax(slice_sums)
        z_minus, z_plus = max(0, best_z - 1), min(mask_volume.shape[2] - 1, best_z + 1)

        img_z_minus, img_z, img_z_plus = image[:, :, z_minus], image[:, :, best_z], image[:, :, z_plus]
        mask_z = (mask_volume[:, :, best_z] > 0).astype(np.int64)

        if self.normalize:
            img_z_minus, img_z, img_z_plus = map(self._normalize_image, (img_z_minus, img_z, img_z_plus))

        img_z_minus = self._resize_2d(img_z_minus)
        img_z = self._resize_2d(img_z)
        img_z_plus = self._resize_2d(img_z_plus)
        mask_z = self._resize_2d(mask_z, is_mask=True)

        image_25d = np.stack([img_z_minus, img_z, img_z_plus], axis=0)

        return {
            "image": torch.from_numpy(image_25d).float(),
            "mask": torch.from_numpy(mask_z).long(),
            "text": sample["text"],
            "patient_id": sample["patient_id"],
            "idx": real_idx
        }