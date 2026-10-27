import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model.dataset import DangerDataset
from src.model.mlp_network import DangerScoreNetwork

def compute_combined_loss(predictions: torch.Tensor, targets: torch.Tensor, ranking_weight: float = 0.5, margin: float = 0.5) -> torch.Tensor:
    """
    Computes Huber regression loss combined with pairwise ranking loss.
    
    Parameters:
        predictions (torch.Tensor): Model danger predictions of shape (batch, 1).
        targets (torch.Tensor): Ground truth ratings of shape (batch, 1).
        ranking_weight (float): Multiplier for the pairwise ranking penalty.
        margin (float): Required score separation between safe and dangerous samples.
    Returns:
        torch.Tensor: Combined scalar loss.
    """
    # Huber loss prevents extreme target values from causing exploding gradients
    regression_loss = F.smooth_l1_loss(predictions, targets, beta=1.0)
    
    # Compare all pairs in current batch to enforce correct relative danger ranking
    diff_targets = targets - targets.T
    diff_preds = predictions - predictions.T
    
    valid_pairs = diff_targets > 0.5
    if valid_pairs.sum() > 0:
        ranking_loss = F.relu(margin - diff_preds)[valid_pairs].mean()
    else:
        ranking_loss = torch.tensor(0.0, device=predictions.device)
        
    return regression_loss + (ranking_weight * ranking_loss)

def init_training_components(embeddings_dir: str) -> tuple:
    """
    Initializes the dataset, dataloader, model, and optimizer.
    
    Parameters:
        embeddings_dir (str): Path to stored embeddings.
    Returns:
        tuple: (dataloader, model, optimizer)
    """
    dataset = DangerDataset(embeddings_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = DangerScoreNetwork()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    
    return dataloader, model, optimizer

def run_training_loop(epochs: int, dataloader: DataLoader, model: nn.Module, optimizer: optim.Optimizer) -> None:
    """
    Executes the training loop over the dataset.
    
    Parameters:
        epochs (int): Number of training epochs.
        dataloader (DataLoader): Data provider.
        model (nn.Module): Neural network.
        optimizer (optim.Optimizer): Optimization algorithm.
    Returns:
        None
    """
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for visual_emb, text_emb, target_score in dataloader:
            optimizer.zero_grad()
            
            predictions = model(visual_emb, text_emb)
            loss = compute_combined_loss(predictions, target_score)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch [{epoch+1}/{epochs}], Combined Loss: {avg_loss:.4f}")

def train_pipeline() -> None:
    """
    Main function connecting data, model initialization, and training loop.
    
    Parameters: None
    Returns: None
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    embeddings_dir = os.path.join(base_dir, "data", "embeddings")
    model_save_dir = os.path.join(base_dir, "data", "models")
    os.makedirs(model_save_dir, exist_ok=True)
    
    dataloader, model, optimizer = init_training_components(embeddings_dir)
    
    if len(dataloader.dataset) == 0:
        print("No valid data found. Run preprocessing and feature extraction first.")
        return
        
    print(f"Starting training on {len(dataloader.dataset)} samples...")
    run_training_loop(epochs=50, dataloader=dataloader, model=model, optimizer=optimizer)
    
    save_path = os.path.join(model_save_dir, "danger_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_pipeline()
