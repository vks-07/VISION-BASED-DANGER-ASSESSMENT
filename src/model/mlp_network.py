import torch
import torch.nn as nn
import torch.nn.functional as F

class DangerScoreNetwork(nn.Module):
    """
    Multimodal Danger Score Network with Bi-LSTM temporal sequence modeling.
    """
    def __init__(self, visual_dim: int = 768, text_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        
        # Bi-LSTM processes the 50 frames to model motion and speed over time
        self.temporal_lstm = nn.LSTM(
            input_size=visual_dim,
            hidden_size=visual_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        
        combined_dim = visual_dim + text_dim
        self.ln_input = nn.LayerNorm(combined_dim)
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, visual_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Computes continuous danger score from visual frame sequence and text context.
        
        Parameters:
            visual_embedding (torch.Tensor): Visual frame vectors of shape (batch, frames, 768) or (batch, 768).
            text_embedding (torch.Tensor): Text description embedding of shape (batch, 768).
        Returns:
            torch.Tensor: Bounded danger score between 0.0 and 10.0.
        """
        # If input is 2D (batch, 768), add sequence dimension
        if visual_embedding.dim() == 2:
            visual_embedding = visual_embedding.unsqueeze(1)
            
        # Bi-LSTM tracks how the danger evolves from frame 0 to frame 49
        lstm_out, _ = self.temporal_lstm(visual_embedding)
        visual_feat = torch.mean(lstm_out, dim=1)
        
        x = torch.cat((visual_feat, text_embedding), dim=1)
        x = self.ln_input(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        score = self.output_layer(x)
        bounded_score = torch.sigmoid(score) * 10.0
        return bounded_score
