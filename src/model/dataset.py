import os
import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Tuple

class DangerDataset(Dataset):
    """
    PyTorch Dataset loading pre-extracted embeddings with temporal data augmentation.
    """
    def __init__(self, embeddings_dir: str, hf_dataset_name: str = "pranked03/ViDAS", split: str = "train", augment: bool = False):
        self.embeddings_dir = embeddings_dir
        self.augment = augment
        self.dataset = load_dataset(hf_dataset_name, split=split)
        
        # Pre-filter dataset to only include rows with successfully generated embeddings
        self.valid_records = []
        for row in self.dataset:
            vid = row['video_id']
            if os.path.exists(os.path.join(embeddings_dir, f"{vid}_visual.npy")) and \
               os.path.exists(os.path.join(embeddings_dir, f"{vid}_text.npy")):
                self.valid_records.append(row)

    def __len__(self) -> int:
        return len(self.valid_records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample from the dataset with optional temporal data augmentation.
        
        Parameters:
            idx (int): Index of the sample.
        Returns:
            tuple: (visual_tensor, text_tensor, rating_tensor)
        """
        item = self.valid_records[idx]
        video_id = item['video_id']
        rating = torch.tensor([item['rating']], dtype=torch.float32)

        visual_path = os.path.join(self.embeddings_dir, f"{video_id}_visual.npy")
        text_path = os.path.join(self.embeddings_dir, f"{video_id}_text.npy")

        visual_emb = np.load(visual_path)
        text_emb = np.load(text_path)

        # Ensure visual embedding has temporal sequence dimension (frames, 768)
        if len(visual_emb.shape) == 1:
            visual_emb = np.expand_dims(visual_emb, axis=0)
            
        visual_tensor = torch.tensor(visual_emb, dtype=torch.float32)
        text_tensor = torch.tensor(np.squeeze(text_emb), dtype=torch.float32)

        # Training augmentation: Temporal frame masking and feature noise
        if self.augment:
            # 1. Add mild sensor camera noise to prevent exact vector memorization
            noise = torch.randn_like(visual_tensor) * 0.015
            visual_tensor = visual_tensor + noise
            
            # 2. Random temporal frame masking: randomly drops 10% of frames to simulate frame drops
            mask = (torch.rand(visual_tensor.shape[0], 1) > 0.1).float()
            visual_tensor = visual_tensor * mask

        return (visual_tensor, text_tensor, rating)
