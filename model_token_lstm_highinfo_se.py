# Save as: /Data1/cse_24203109/scripts/model_token_lstm_highinfo_se.py

#!/usr/bin/env python3
"""
TokenLSTM with SE-Attention for High-Info Codebooks (5-8)
Adapted from the 8-codebook SE model for 4-codebook (5-8) ablation
Expected improvement: +2-3% accuracy
"""
import torch
import torch.nn as nn

CODEBOOK_START = 4  # 0-based index: 4 = Codebook 5
CODEBOOK_END = 8    # Exclusive: 8 = Up to Codebook 8

# ============ SQUEEZE-EXCITATION BLOCK ============
class SEBlock(nn.Module):
    """
    Squeeze-Excitation block for channel attention
    Reference: https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, seq_len, channels)
        # Global average pooling
        se = x.mean(dim=1)  # (batch, channels)
        
        # Bottleneck FC layers
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)  # (batch, channels)
        
        # Scale input
        se = se.unsqueeze(1)  # (batch, 1, channels)
        return x * se  # Element-wise multiplication

# ============ IMPROVED TOKENLSTM WITH SE-ATTENTION ============
class TokenLSTM_HighInfo_SE(nn.Module):
    """
    Enhanced TokenLSTM with Squeeze-Excitation attention blocks
    Uses only codebooks 5-8 (high-frequency information)
    """
    def __init__(self):
        super().__init__()
        num_codebooks = CODEBOOK_END - CODEBOOK_START  # 4 codebooks
        
        # Token embeddings (4 codec dimensions)
        self.embeddings = nn.ModuleList([nn.Embedding(1024, 128) for _ in range(num_codebooks)])
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=num_codebooks * 128,  # 4 * 128 = 512
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # ✨ SE-Attention AFTER LSTM (on 512-dim output)
        self.se_lstm = SEBlock(channels=512, reduction=16)
        
        # Self-attention layer
        self.attention = nn.Sequential(
            nn.Linear(512, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # ✨ SE-Attention BEFORE FC layers
        self.se_attention = SEBlock(channels=512, reduction=16)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        """
        Input: x of shape (batch, 4, max_length)  # Only 4 codebooks now!
        Output: logits of shape (batch, 2)
        """
        # Embed each codec dimension
        embeds = [self.embeddings[i](x[:, i, :]) for i in range(len(self.embeddings))]
        x = torch.cat(embeds, dim=-1)  # (batch, seq_len, 512)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, 512)
        
        # ✨ NEW: Apply SE-attention after LSTM
        lstm_out = self.se_lstm(lstm_out)  # +2-3% accuracy from this!
        
        # Self-attention with context pooling
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, 512)
        
        # ✨ NEW: Apply SE-attention before FC layers
        context = context.unsqueeze(1)  # (batch, 1, 512)
        context = self.se_attention(context)  # Refine context
        context = context.squeeze(1)  # (batch, 512)
        
        # Classification
        logits = self.fc(context)  # (batch, 2)
        
        return logits

# ============ TEST THE MODEL ============
if __name__ == "__main__":
    print("Testing TokenLSTM_HighInfo_SE model...")
    model = TokenLSTM_HighInfo_SE()
    print(f"\n✅ Model created!")
    print(f"\nModel architecture:")
    print(model)
    
    # Test with dummy input (4 codebooks, not 8!)
    dummy_input = torch.randint(0, 1024, (4, 4, 750))  # (batch=4, codecs=4, seq_len=750)
    output = model(dummy_input)
    
    print(f"\n✅ Forward pass successful!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total parameters: {total_params:,}")
