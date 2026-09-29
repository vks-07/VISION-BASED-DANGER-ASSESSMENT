import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DangerDataset
from mlp_network import DangerScoreNetwork

def init_training_components(embeddings_dir: str) -> tuple:
    """
    Initializes the dataset, dataloader, model, loss function, and optimizer.
    
    Parameters:
        embeddings_dir (str): Path to stored embeddings.
    Returns:
        tuple: (dataloader, model, criterion, optimizer)
    """
    dataset = DangerDataset(embeddings_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = DangerScoreNetwork()
    
    # MSE is the standard loss function for continuous regression tasks, 
    # measuring the squared distance between predicted rating and actual rating.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    return dataloader, model, criterion, optimizer

def run_training_loop(epochs: int, dataloader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer) -> None:
    """
    Executes the training loop over the dataset.
    
    Parameters:
        epochs (int): Number of training epochs.
        dataloader (DataLoader): Data provider.
        model (nn.Module): Neural network.
        criterion (nn.Module): Loss function.
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
            loss = criterion(predictions, target_score)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch [{epoch+1}/{epochs}], MSE Loss: {avg_loss:.4f}")

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
    
    dataloader, model, criterion, optimizer = init_training_components(embeddings_dir)
    
    if len(dataloader.dataset) == 0:
        print("No valid data found. Run preprocessing and feature extraction first.")
        return
        
    print(f"Starting training on {len(dataloader.dataset)} samples...")
    run_training_loop(epochs=10, dataloader=dataloader, model=model, criterion=criterion, optimizer=optimizer)
    
    save_path = os.path.join(model_save_dir, "danger_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_pipeline()
