#!/usr/bin/env python3
import torch
import torch.nn as nn

CODEBOOK_START = 4
CODEBOOK_END   = 8


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1     = nn.Linear(channels, max(1, channels // reduction))
        self.fc2     = nn.Linear(max(1, channels // reduction), channels)
        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = x.mean(dim=1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        se = se.unsqueeze(1)
        return x * se


class TokenLSTM_HighInfo_SE(nn.Module):
    def __init__(self):
        super().__init__()
        num_codebooks = CODEBOOK_END - CODEBOOK_START

        self.embeddings = nn.ModuleList([nn.Embedding(1024, 128) for _ in range(num_codebooks)])

        self.lstm = nn.LSTM(
            input_size=num_codebooks * 128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.se_lstm = SEBlock(channels=512, reduction=16)

        self.attention = nn.Sequential(
            nn.Linear(512, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.se_attention = SEBlock(channels=512, reduction=16)

        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        embeds   = [self.embeddings[i](x[:, i, :]) for i in range(len(self.embeddings))]
        x        = torch.cat(embeds, dim=-1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.se_lstm(lstm_out)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context  = torch.sum(lstm_out * attn_weights, dim=1)
        context  = context.unsqueeze(1)
        context  = self.se_attention(context)
        context  = context.squeeze(1)
        logits   = self.fc(context)
        return logits


if __name__ == "__main__":
    print("Testing TokenLSTM_HighInfo_SE model...")
    model = TokenLSTM_HighInfo_SE()
    print(f"\nModel created!")
    print(f"\nModel architecture:")
    print(model)

    dummy_input = torch.randint(0, 1024, (4, 4, 750))
    output = model(dummy_input)

    print(f"\nForward pass successful!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
