import torch
import torch.nn as nn
import torch.nn.functional as F

class DangerScoreNetwork(nn.Module):
    """
    Continuous Regression Network predicting danger scores from concatenated visual and text embeddings.
    """
    def __init__(self, visual_dim: int = 768, text_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        
        combined_dim = visual_dim + text_dim
        
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        
        # Final layer predicts a single continuous float value
        self.output_layer = nn.Linear(hidden_dim // 4, 1)

    def forward(self, visual_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Parameters:
            visual_embedding (torch.Tensor): Visual features, shape (batch_size, 768).
            text_embedding (torch.Tensor): Text features, shape (batch_size, 768).
        Returns:
            torch.Tensor: Predicted danger score, shape (batch_size, 1).
        """
        # Concatenate features along the column dimension
        x = torch.cat((visual_embedding, text_embedding), dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        score = self.output_layer(x)
        
        # We use a Sigmoid function multiplied by 5.0. 
        # This acts as a mathematical guarantee that the network can never 
        # accidentally predict a negative score or a score higher than 5.0, 
        # perfectly matching the ViDAS dataset's rating scale.
        bounded_score = torch.sigmoid(score) * 5.0
        
        return bounded_score
