#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import sys

sys.path.insert(0, '/Data1/cse_24203109/scripts')
from model_token_lstm_se import TokenLSTM_SE

print("=" * 80)
print("ENSEMBLE EVALUATION WITH SE-ATTENTION (4 Models on CoSG)")
print("=" * 80)

MODELS_DIR   = "/Data1/cse_24203109/models/ensemble_4_with_se"
COSG_CSV     = "/Data1/cse_24203109/processed_data/cosg_eval.csv"
COSG_TOKENS  = "/Data1/cse_24203109/processed_data/tokens_cosg"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

selected_models = [
    'HiFi_Codec_16k_320d_large_universal',
    'bigcodec',
    'FunCodec_en-libritts-16k-gr1nq32ds320',
    'FACodec_encodec-decoder-v2_16k',
]

print(f"\nSELECTED MODELS (4 total - WITH SE-ATTENTION):")
print(f"\n  TIME-DOMAIN (3):")
for i, model in enumerate(selected_models[:3], 1):
    print(f"     T{i}. {model}")
print(f"\n  FREQUENCY-DOMAIN (1):")
print(f"     F1. {selected_models[3]}")

print(f"\n[1/3] Loading models...")
models = {}
for model_name in tqdm(selected_models, desc="Loading"):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    if os.path.exists(model_path):
        model = TokenLSTM_SE().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        models[model_name] = model

print(f"Loaded {len(models)}/4 models")

print(f"\n[2/3] Loading CoSG dataset...")
df = pd.read_csv(COSG_CSV)
print(f"   Total: {len(df):,} samples")
print(f"   Bonafide: {(df['label'] == 0).sum():,}")
print(f"   Spoof: {(df['label'] == 1).sum():,}")

all_preds       = []
all_labels      = []
all_confidences = []

print(f"\n[3/3] Running inference...")
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
    token_path = os.path.join(COSG_TOKENS, f"{os.path.splitext(row['filename'])[0]}.npy")

    try:
        tokens = np.load(token_path)
        max_length = 750
        if tokens.shape[1] > max_length:
            tokens = tokens[:, :max_length]
        else:
            pad_len = max_length - tokens.shape[1]
            tokens = np.pad(tokens, ((0, 0), (0, pad_len)), mode='constant')

        tokens_tensor = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)

        weights = {
            'HiFi_Codec_16k_320d_large_universal': 0.7501,
            'bigcodec': 0.7301,
            'FunCodec_en-libritts-16k-gr1nq32ds320': 0.6427,
            'FACodec_encodec-decoder-v2_16k': 0.5899,
        }

        weighted_score = 0
        total_weight   = 0

        with torch.no_grad():
            for model_name, model in models.items():
                weight    = weights[model_name]
                outputs   = model(tokens_tensor)
                probs     = torch.softmax(outputs, dim=1)
                prob_spoof = probs[0, 1].item()
                weighted_score += weight * prob_spoof
                total_weight   += weight

        final_score = weighted_score / total_weight if total_weight > 0 else 0.5

        all_confidences.append(final_score)
        all_labels.append(row['label'])

    except:
        all_confidences.append(0.5)
        all_labels.append(row['label'])

print(f"\n{'=' * 80}")
print(f"CALCULATING OPTIMAL THRESHOLD...")
print(f"{'=' * 80}\n")

fpr, tpr, thresholds = roc_curve(all_labels, all_confidences)
roc_auc = auc(fpr, tpr)

youdens       = tpr - fpr
optimal_idx   = np.argmax(youdens)
optimal_threshold = thresholds[optimal_idx]

print(f"THRESHOLD COMPARISON:")
print(f"   Default threshold (0.50): {len([1 for x in all_confidences if x > 0.5])}/{len(all_confidences)} predicted as spoof")
print(f"   Optimal threshold ({optimal_threshold:.4f}): {len([1 for x in all_confidences if x > optimal_threshold])}/{len(all_confidences)} predicted as spoof")

all_preds = np.array([1 if conf > optimal_threshold else 0 for conf in all_confidences])

print(f"\n{'=' * 80}")
print(f"FINAL ENSEMBLE RESULTS (WITH SE-ATTENTION)")
print(f"{'=' * 80}\n")

acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec  = recall_score(all_labels, all_preds)
f1   = f1_score(all_labels, all_preds)

print(f"BASIC METRICS:")
print(f"   Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
print(f"   Precision: {prec:.4f}")
print(f"   Recall:    {rec:.4f}")
print(f"   F1-Score:  {f1:.4f}")

correct_bonafide = sum(1 for true, pred in zip(all_labels, all_preds) if true == 0 and pred == 0)
correct_spoof    = sum(1 for true, pred in zip(all_labels, all_preds) if true == 1 and pred == 1)
total_bonafide   = sum(1 for true in all_labels if true == 0)
total_spoof      = sum(1 for true in all_labels if true == 1)

bonafide_acc = correct_bonafide / total_bonafide if total_bonafide > 0 else 0
spoof_acc    = correct_spoof / total_spoof if total_spoof > 0 else 0

print(f"\nPER-CLASS ACCURACY:")
print(f"   Bonafide Detection: {bonafide_acc:.4f} ({correct_bonafide}/{total_bonafide})")
print(f"   Spoof Detection:    {spoof_acc:.4f} ({correct_spoof}/{total_spoof})")

cm = confusion_matrix(all_labels, all_preds)
print(f"\nCONFUSION MATRIX:")
print(f"           Pred Bonafide  Pred Spoof")
print(f"Real Bonafide      {cm[0][0]:5d}        {cm[0][1]:5d}")
print(f"Real Spoof         {cm[1][0]:5d}        {cm[1][1]:5d}")

print(f"\n{'=' * 80}")
print(f"EER ANALYSIS (Equal Error Rate)")
print(f"{'=' * 80}\n")

fnr = 1 - tpr
eer = (fpr[optimal_idx] + fnr[optimal_idx]) / 2

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"EER (Equal Error Rate): {eer:.4f} ({eer * 100:.2f}%)")
print(f"   Optimal Threshold: {optimal_threshold:.4f}")
print(f"   FPR at EER: {fpr[optimal_idx]:.4f}")
print(f"   FNR at EER: {fnr[optimal_idx]:.4f}")
print(f"   Youden Index: {youdens[optimal_idx]:.4f}")

print(f"\n{'=' * 80}")
print(f"COMPARISON")
print(f"{'=' * 80}\n")

print(f"{'Approach':<55} {'Accuracy':<12} {'EER':<10}")
print(f"-" * 80)
print(f"{'DAC24 baseline':<55} {'48.19%':<12} {'~51.81%':<10}")
print(f"{'Previous (3T+1F, Focal+Aug, 15 epochs)':<55} {'77.52%':<12} {'22.43%':<10}")
print(f"{'NEW (3T+1F, SE-Attention, 15 epochs)':<55} {f'{acc * 100:.2f}%':<12} {f'{eer * 100:.2f}%':<10}")
print(f"{'Paper Best (W2V2-AASIST DEC-balance)':<55} {'88.00%':<12} {'11.91%':<10}")

improvement_acc = ((acc - 0.4819) / 0.4819) * 100
improvement_eer = ((0.5181 - eer) / 0.5181) * 100 if eer > 0 else 0

print(f"\nIMPROVEMENT OVER DAC24:")
print(f"   Accuracy improvement: +{improvement_acc:.1f}%")
print(f"   EER improvement: +{improvement_eer:.1f}%")

print(f"\n{'=' * 80}")
print(f"SE-ATTENTION ENSEMBLE EVALUATION COMPLETE!")
print(f"{'=' * 80}\n")
