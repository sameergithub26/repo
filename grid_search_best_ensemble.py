#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools

print("="*80)
print("🎵 GRID SEARCH: FINDING OPTIMAL ENSEMBLE COMBINATION")
print("="*80)

MODELS_DIR = "/Data1/cse_24203109/models/ensemble_29_codecs_proper"
COSG_CSV = "/Data1/cse_24203109/processed_data/cosg_eval.csv"
COSG_TOKENS = "/Data1/cse_24203109/processed_data/tokens_cosg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TokenLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(1024, 128) for _ in range(8)])
        self.lstm = nn.LSTM(1024, 256, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = nn.Sequential(
            nn.Linear(512, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        embeds = [self.embeddings[i](x[:, i, :]) for i in range(8)]
        x = torch.cat(embeds, dim=-1)
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn, dim=1)
        return self.fc(context)

# Pre-ranked models by performance
time_domain = [
    ('HiFi_Codec_16k_320d_large_universal', 0.7501),
    ('bigcodec', 0.7301),
    ('FunCodec_en-libritts-16k-gr1nq32ds320', 0.6427),
    ('llm_codec', 0.5949),
    ('vocos_encodec_12', 0.5492),
]

freq_domain = [
    ('FACodec_encodec-decoder-v2_16k', 0.5899),
    ('vocos_encodec_6', 0.5698),
    ('spectralcodecs', 0.5303),
    ('snac_24khz', 0.5292),
    ('snac_44khz', 0.4853),
]

print(f"\n📊 TIME-DOMAIN models (top 5):")
for i, (name, acc) in enumerate(time_domain, 1):
    print(f"   T{i}: {name} ({acc:.4f})")

print(f"\n📊 FREQUENCY-DOMAIN models (top 5):")
for i, (name, acc) in enumerate(freq_domain, 1):
    print(f"   F{i}: {name} ({acc:.4f})")

# Load CoSG data
print(f"\n[1/2] Loading CoSG dataset...")
df = pd.read_csv(COSG_CSV)
print(f"   Loaded {len(df):,} samples")

# Test combinations: T1-T5 vs F1-F5
results = []

combinations_to_test = [
    # (top_n_time, top_n_freq, description)
    (1, 1, "Top-1 Time + Top-1 Freq"),
    (1, 2, "Top-1 Time + Top-2 Freq"),
    (2, 1, "Top-2 Time + Top-1 Freq"),
    (2, 2, "Top-2 Time + Top-2 Freq"),  # Already tested
    (2, 3, "Top-2 Time + Top-3 Freq"),
    (3, 1, "Top-3 Time + Top-1 Freq"),
    (3, 2, "Top-3 Time + Top-2 Freq"),
    (3, 3, "Top-3 Time + Top-3 Freq"),  # Already tested
    (4, 1, "Top-4 Time + Top-1 Freq"),
    (4, 2, "Top-4 Time + Top-2 Freq"),
]

print(f"\n[2/2] Testing {len(combinations_to_test)} combinations...")

for top_t, top_f, desc in combinations_to_test:
    # Select models
    selected_models = {}
    selected_models.update(dict(time_domain[:top_t]))
    selected_models.update(dict(freq_domain[:top_f]))
    
    # Load models
    models = {}
    for codec_name in selected_models.keys():
        model_path = os.path.join(MODELS_DIR, f"{codec_name}.pth")
        if os.path.exists(model_path):
            model = TokenLSTM().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            models[codec_name] = model
    
    if len(models) != (top_t + top_f):
        print(f"⚠️  {desc}: Only {len(models)}/{top_t + top_f} models found, skipping...")
        continue
    
    # Inference
    all_preds = []
    all_labels = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc, leave=False):
        token_path = os.path.join(COSG_TOKENS, f"{os.path.splitext(row['filename'])[0]}.npy")
        
        try:
            tokens = np.load(token_path)
            max_length = 750
            if tokens.shape[1] > max_length:
                tokens = tokens[:, :max_length]
            else:
                pad_len = max_length - tokens.shape[1]
                tokens = np.pad(tokens, ((0,0), (0,pad_len)), mode='constant')
            
            tokens_tensor = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)
            
            weighted_score = 0
            total_weight = 0
            
            with torch.no_grad():
                for codec_name, model in models.items():
                    weight = selected_models[codec_name]
                    outputs = model(tokens_tensor)
                    prob_spoof = torch.softmax(outputs, dim=1)[0, 1].item()
                    weighted_score += weight * prob_spoof
                    total_weight += weight
            
            final_score = weighted_score / total_weight if total_weight > 0 else 0.5
            ensemble_pred = 1 if final_score > 0.5 else 0
            
            all_preds.append(ensemble_pred)
            all_labels.append(row['label'])
        
        except:
            all_preds.append(0)
            all_labels.append(row['label'])
    
    # Evaluate
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    results.append({
        'combination': desc,
        'n_models': top_t + top_f,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })
    
    print(f"✓ {desc:40s} | Acc: {acc:.4f} | F1: {f1:.4f}")

# Sort by accuracy
results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)

print(f"\n{'='*80}")
print(f"🏆 TOP 5 COMBINATIONS")
print(f"{'='*80}\n")
print(f"{'Combination':<45} {'Models':<8} {'Accuracy':<12} {'F1-Score':<10}")
print("-" * 85)

for i, r in enumerate(results_sorted[:5], 1):
    print(f"{i}. {r['combination']:<42} {r['n_models']:<8} {r['accuracy']:.4f}      {r['f1']:.4f}")

print(f"\n{'='*80}")
print(f"✅ GRID SEARCH COMPLETE!")
print(f"{'='*80}")
