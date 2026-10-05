import os
import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Tuple

class DangerDataset(Dataset):
    """
    PyTorch Dataset loading pre-extracted embeddings and ground truth ratings.
    """
    def __init__(self, embeddings_dir: str, hf_dataset_name: str = "pranked03/ViDAS", split: str = "train"):
        self.embeddings_dir = embeddings_dir
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
        Retrieves a single sample from the dataset.
        
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

        # Temporal Pooling: Average frames to a 1D vector (768,)
        if len(visual_emb.shape) == 2:
            visual_emb = np.mean(visual_emb, axis=0)
            
        text_emb = np.squeeze(text_emb)

        return (
            torch.tensor(visual_emb, dtype=torch.float32), 
            torch.tensor(text_emb, dtype=torch.float32), 
            rating
        )
