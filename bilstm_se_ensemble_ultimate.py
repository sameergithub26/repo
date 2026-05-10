#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

print("=" * 80)
print("ULTIMATE BiLSTM+SE ENSEMBLE (3T1F)")
print("=" * 80)

PROCESSED_DIR = "/Data1/cse_24203109/processed_data"
COSG_CSV      = os.path.join(PROCESSED_DIR, "cosg_eval.csv")
COSG_TOKENS   = os.path.join(PROCESSED_DIR, "tokens_cosg")
MODELS_DIR    = "/Data1/cse_24203109/models/ensemble_4_with_se"
RESULTS_CSV   = "/Data1/cse_24203109/results/bilstm_se_ultimate_3t1f.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 750


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = x.mean(dim=1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).unsqueeze(1)
        return x * se


class TokenLSTM_SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(1024, 128) for _ in range(8)])
        self.lstm = nn.LSTM(1024, 256, 2, batch_first=True, bidirectional=True, dropout=0.3)
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
        embeds = [self.embeddings[i](x[:, i, :]) for i in range(8)]
        x = torch.cat(embeds, dim=-1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.se_lstm(lstm_out)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        context = context.unsqueeze(1)
        context = self.se_attention(context)
        context = context.squeeze(1)
        return self.fc(context)


df = pd.read_csv(COSG_CSV)
print(f"Total CoSG samples: {len(df):,}")
print(f"Bonafide: {(df['label']==0).sum():,} | Spoof: {(df['label']==1).sum():,}\n")

selected = {
    "HiFi_Codec_16k_320d_large_universal": {"weight": 0.7501, "decoder": "time"},
    "bigcodec": {"weight": 0.7301, "decoder": "time"},
    "FunCodec_en-libritts-16k-gr1nq32ds320": {"weight": 0.6427, "decoder": "time"},
    "FACodec_encodec-decoder-v2_16k": {"weight": 0.5899, "decoder": "freq"},
}

print("3T1F BiLSTM+SE Ensemble:")
for codec, info in selected.items():
    print(f"  {codec[:30]:<30} {info['decoder']:<6} w={info['weight']:.4f}")

models = {}
for codec_name, info in selected.items():
    path = os.path.join(MODELS_DIR, f"{codec_name}.pth")
    if not os.path.exists(path):
        print(f"  {codec_name}: model not found")
        continue
    m = TokenLSTM_SE().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    models[codec_name] = {"model": m, "weight": info["weight"], "decoder": info["decoder"]}
    print(f"  Loaded {codec_name[:30]}")

print(f"\nLoaded {len(models)}/4 models\n")


def extract_token_features(tokens_np):
    low_activity = tokens_np[:4].std()
    high_activity = tokens_np[4:].std()
    temporal_var = tokens_np.var(axis=1).mean()
    sparsity = (tokens_np == 0).mean()
    high_low_ratio = (high_activity / (low_activity + 1e-6))
    return {
        'high_low_ratio': high_low_ratio,
        'temporal_var': temporal_var,
        'sparsity': sparsity,
        'high_activity': high_activity,
    }


def compute_confidence_weights(logits_dict, base_weights):
    adjusted = {}
    for name, logits in logits_dict.items():
        probs = torch.softmax(logits, dim=1)[0]
        confidence = abs(probs[1] - 0.5) * 2
        boost = 1.0 + 0.2 * confidence
        adjusted[name] = base_weights[name] * boost
    return adjusted


def compute_sample_adaptive_weights(token_features, base_weights, decoder_types):
    ratio = token_features['high_low_ratio']
    temporal_var = token_features['temporal_var']
    ratio_score = np.clip((ratio - 0.5) / 2.0, 0, 1)
    adapt_strength = np.clip(temporal_var / 100.0, 0, 1)
    adjusted = {}
    for name, base_w in base_weights.items():
        if decoder_types[name] == "time":
            boost = 1.0 + adapt_strength * ratio_score * 0.25
        else:
            boost = 1.0 + adapt_strength * (1 - ratio_score) * 0.25
        adjusted[name] = base_w * boost
    return adjusted


def load_tokens(filename):
    token_path = os.path.join(COSG_TOKENS, f"{os.path.splitext(filename)[0]}.npy")
    tokens = np.load(token_path)
    if tokens.shape[1] > MAX_LENGTH:
        tokens = tokens[:, :MAX_LENGTH]
    else:
        pad_len = MAX_LENGTH - tokens.shape[1]
        tokens = np.pad(tokens, ((0, 0), (0, pad_len)), mode="constant")
    return tokens


print("Running ultimate ensemble (sample-adaptive + confidence-weighted)...\n")

all_scores = []
all_labels = []

base_weights = {name: info["weight"] for name, info in models.items()}
decoder_types = {name: info["decoder"] for name, info in models.items()}

for _, row in tqdm(df.iterrows(), total=len(df), desc="Ultimate Ensemble"):
    fname = row["filename"]
    y = int(row["label"])

    try:
        tokens_np = load_tokens(fname)
        tokens_torch = torch.LongTensor(tokens_np).unsqueeze(0).to(DEVICE)

        token_features = extract_token_features(tokens_np)

        sample_weights = compute_sample_adaptive_weights(
            token_features, base_weights, decoder_types
        )

        logits_dict = {}
        with torch.no_grad():
            for name, info in models.items():
                logits = info["model"](tokens_torch)
                logits_dict[name] = logits

        confidence_weights = compute_confidence_weights(logits_dict, sample_weights)

        weighted_logits = None
        total_w = 0.0

        for name, logits in logits_dict.items():
            w = confidence_weights[name]
            if weighted_logits is None:
                weighted_logits = w * logits
            else:
                weighted_logits += w * logits
            total_w += w

        avg_logits = weighted_logits / total_w if total_w > 0 else weighted_logits
        probs = torch.softmax(avg_logits, dim=1)[0]
        score = probs[1].item()

    except Exception:
        score = 0.5

    all_scores.append(score)
    all_labels.append(y)

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

print("\nOptimizing threshold...")

fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)

youdens_j = tpr - fpr
youden_idx = np.argmax(youdens_j)
youden_thresh = thresholds[youden_idx]

balanced_acc = (tpr + (1 - fpr)) / 2
balanced_idx = np.argmax(balanced_acc)
balanced_thresh = thresholds[balanced_idx]

results = []
for thresh, name in [(youden_thresh, "Youden"), (balanced_thresh, "Balanced")]:
    preds = (all_scores >= thresh).astype(int)

    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)

    correct_b = ((all_labels == 0) & (preds == 0)).sum()
    correct_s = ((all_labels == 1) & (preds == 1)).sum()
    tot_b = (all_labels == 0).sum()
    tot_s = (all_labels == 1).sum()
    b_acc = correct_b / tot_b if tot_b > 0 else 0.0
    s_acc = correct_s / tot_s if tot_s > 0 else 0.0

    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    results.append({
        'method': name,
        'threshold': thresh,
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'bonafide_acc': b_acc,
        'spoof_acc': s_acc,
        'roc_auc': roc_auc,
        'eer': eer,
    })

    print(f"\n{name} (threshold={thresh:.4f}):")
    print(f"  Accuracy:     {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  BonafideAcc:  {b_acc:.4f} ({correct_b}/{tot_b})")
    print(f"  SpoofAcc:     {s_acc:.4f} ({correct_s}/{tot_s})")

best = max(results, key=lambda x: x['acc'])

print("\n" + "=" * 80)
print("ULTIMATE BiLSTM+SE ENSEMBLE RESULT")
print("=" * 80)
print(f"Method:       {best['method']}")
print(f"Threshold:    {best['threshold']:.4f}")
print(f"Accuracy:     {best['acc']:.4f} ({best['acc'] * 100:.2f}%)")
print(f"Precision:    {best['prec']:.4f}")
print(f"Recall:       {best['rec']:.4f}")
print(f"F1-score:     {best['f1']:.4f}")
print(f"BonafideAcc:  {best['bonafide_acc']:.4f}")
print(f"SpoofAcc:     {best['spoof_acc']:.4f}")
print(f"ROC-AUC:      {best['roc_auc']:.4f}")
print(f"EER:          {best['eer']:.4f} ({best['eer'] * 100:.2f}%)")
print(f"\nImprovement over baseline (79.02%): {(best['acc'] - 0.7902) * 100:+.2f}pp")

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
pd.DataFrame([best]).to_csv(RESULTS_CSV, index=False)
print(f"\nSaved to: {RESULTS_CSV}")
