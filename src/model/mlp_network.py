import torch
import torch.nn as nn
import torch.nn.functional as F

class DangerScoreNetwork(nn.Module):
    """
    Multimodal Danger Score Network with Bi-LSTM temporal modeling and Cross-Attention fusion.
    """
    def __init__(self, visual_dim: int = 768, text_dim: int = 768, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        
        # Bi-LSTM processes the 50 frames to model motion and speed over time
        self.temporal_lstm = nn.LSTM(
            input_size=visual_dim,
            hidden_size=visual_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        
        # Cross-Attention: Text queries the visual frame sequence to find hazard moments
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        combined_dim = visual_dim + text_dim
        self.ln_input = nn.LayerNorm(combined_dim)
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, visual_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Computes danger score using temporal modeling and cross-attention between text and video.
        
        Parameters:
            visual_embedding (torch.Tensor): Visual frame vectors of shape (batch, frames, 768) or (batch, 768).
            text_embedding (torch.Tensor): Text description embedding of shape (batch, 768).
        Returns:
            torch.Tensor: Bounded danger score between 0.0 and 10.0.
        """
        # Automatically handle unbatched and batched inputs
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)
            
        if visual_embedding.dim() == 2:
            if visual_embedding.shape[0] == text_embedding.shape[0]:
                visual_embedding = visual_embedding.unsqueeze(1)
            else:
                visual_embedding = visual_embedding.unsqueeze(0)
            
        # 1. Bi-LSTM captures motion patterns across the video frames
        lstm_out, _ = self.temporal_lstm(visual_embedding)
        
        # 2. Cross-Attention: Text description acts as Query to find relevant video frames
        query = text_embedding.unsqueeze(1)
        attended_frames, _ = self.cross_attention(query=query, key=lstm_out, value=lstm_out)
        attended_visual = attended_frames.squeeze(1)
        
        # Residual fusion: combine attended visual features with text context
        fused_text = text_embedding + attended_visual
        visual_summary = torch.mean(lstm_out, dim=1)
        
        x = torch.cat((visual_summary, fused_text), dim=1)
        x = self.ln_input(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        score = self.output_layer(x)
        bounded_score = torch.sigmoid(score) * 10.0
        return bounded_score
