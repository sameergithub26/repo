#!/usr/bin/env python3
"""
YOUR TOKEN-BASED LSTM APPROACH - Training one codec at a time
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

if len(sys.argv) < 2:
    print("Usage: python train_tokens_lstm.py <codec_name>")
    sys.exit(1)

CODEC_NAME = sys.argv[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*80)
print(f"TRAINING TOKEN-BASED LSTM: {CODEC_NAME}")
print("="*80)

# Paths
CSV_FILE = f"/Data1/cse_24203109/datasets_by_codec/{CODEC_NAME}.csv"
TOKEN_DIR = "/Data1/cse_24203109/processed_data/tokens_1M_real_aug"
MODEL_SAVE_DIR = "/Data1/cse_24203109/models/token_lstm_separate"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Config
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

print(f"\n📋 Configuration:")
print(f"  Device: {DEVICE.upper()}")
print(f"  Codec: {CODEC_NAME}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Feature: EnCodec Tokens (8 quantizers)")

# Load data
df = pd.read_csv(CSV_FILE)
print(f"\n✓ Loaded {len(df):,} samples")
print(f"  Bonafide: {(df['label']=='bonafide').sum():,}")
print(f"  Spoof: {(df['label']=='spoof').sum():,}")

# Token dataset
class TokenDataset(Dataset):
    def __init__(self, df, token_dir, max_length=750):
        self.df = df.reset_index(drop=True)
        self.token_dir = token_dir
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Build token filename
        base_filename = os.path.splitext(row['filename'])[0]
        token_filename = f"{base_filename}_orig.npy"
        token_path = os.path.join(self.token_dir, token_filename)
        
        try:
            tokens = np.load(token_path)
            
            # Pad/truncate to max_length
            if tokens.shape[1] > self.max_length:
                tokens = tokens[:, :self.max_length]
            else:
                pad_length = self.max_length - tokens.shape[1]
                tokens = np.pad(tokens, ((0, 0), (0, pad_length)), mode='constant')
            
            tokens = torch.LongTensor(tokens)
            label = 0 if row['label'] == 'bonafide' else 1
            
            return tokens, label
        
        except Exception as e:
            tokens = torch.zeros((8, self.max_length), dtype=torch.long)
            label = 0 if row['label'] == 'bonafide' else 1
            return tokens, label

# LSTM Model with attention
class TokenLSTMClassifier(nn.Module):
    def __init__(self, vocab_size=1024, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Embeddings for 8 quantizers
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for _ in range(8)
        ])
        
        # LSTM
        self.lstm = nn.LSTM(
            embed_dim * 8,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
    
    def forward(self, tokens):
        batch_size, num_quantizers, seq_len = tokens.shape
        
        # Embed each quantizer
        embeds = []
        for i in range(num_quantizers):
            embed = self.embeddings[i](tokens[:, i, :])
            embeds.append(embed)
        
        # Concatenate embeddings
        x = torch.cat(embeds, dim=-1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention pooling
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Classification
        output = self.classifier(context)
        return output

# Prepare data
dataset = TokenDataset(df, TOKEN_DIR)

train_size = int(len(dataset) * TRAIN_SPLIT)
val_size = int(len(dataset) * VAL_SPLIT)
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"\n📊 Data Split:")
print(f"  Train: {len(train_ds):,} samples")
print(f"  Val: {len(val_ds):,} samples")
print(f"  Test: {len(test_ds):,} samples")

# Training
print(f"\n[TRAINING TOKEN-BASED LSTM]")
print("-"*80)

model = TokenLSTMClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

best_val_acc = 0
best_model_state = None
patience_counter = 0

for epoch in range(EPOCHS):
    # TRAIN
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]", leave=False)
    for tokens, labels in pbar:
        tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(tokens)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    
    train_loss /= len(train_loader)
    train_acc = accuracy_score(train_labels, train_preds)
    
    # VALIDATE
    model.eval()
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [VAL]", leave=False)
        for tokens, labels in pbar:
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
            outputs = model(tokens)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_acc = accuracy_score(val_labels, val_preds)
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        print(f"           ✓ New best model (Val Acc: {val_acc:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
    
    scheduler.step(val_acc)
    
    if patience_counter >= 3:
        print(f"Early stopping triggered")
        break

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# TEST
print(f"\n[TESTING]")
print("-"*80)

model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for tokens, labels in tqdm(test_loader, desc="Testing"):
        tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
        outputs = model(tokens)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(test_labels, test_preds)
cm = confusion_matrix(test_labels, test_preds)

print(f"\n✅ RESULTS FOR {CODEC_NAME}:")
print(f"  Best Val Accuracy: {best_val_acc:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"           Pred Bonafide  Pred Spoof")
print(f"  True Bonafide  {cm[0][0]:>8}  {cm[0][1]:>8}")
print(f"  True Spoof     {cm[1][0]:>8}  {cm[1][1]:>8}")

# Save model
model_path = os.path.join(MODEL_SAVE_DIR, f"model_{CODEC_NAME}.pth")
torch.save(model.state_dict(), model_path)
print(f"\n✓ Model saved: {model_path}")

print("\n" + "="*80)
