import torch
import torch.nn as nn
import torch.nn.functional as F

class DangerScoreNetwork(nn.Module):
    def __init__(self, visual_dim: int = 768, text_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        
        combined_dim = visual_dim + text_dim
        
        self.ln_input = nn.LayerNorm(combined_dim)
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, visual_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        x = torch.cat((visual_embedding, text_embedding), dim=1)
        x = self.ln_input(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        score = self.output_layer(x)
        bounded_score = torch.sigmoid(score) * 10.0
        return bounded_score
