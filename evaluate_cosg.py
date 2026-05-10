import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, classification_report
)

print("="*80)
print("🔬 COSG CROSS-DATASET EVALUATION")
print("="*80)

# Paths
MODEL_PATH = "/Data1/cse_24203109/models/1M_models/best_model_1M.pth"
CSV_FILE = "/Data1/cse_24203109/processed_data/cosg_eval.csv"
TOKEN_DIR = "/Data1/cse_24203109/processed_data/tokens_cosg"
RESULTS_DIR = "/Data1/cse_24203109/results/cosg_evaluation"

os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n💻 Device: {DEVICE}")

# ============ MODEL ============
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(1024, 256) for _ in range(8)])
        self.lstm = nn.LSTM(256*8, 512, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = nn.Sequential(nn.Linear(1024, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)
        )
    
    def forward(self, x):
        embeds = [self.embeddings[i](x[:, i, :]) for i in range(8)]
        x = torch.cat(embeds, dim=-1)
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn, dim=1)
        return self.classifier(context)

# ============ DATASET ============
class CoSGDataset(Dataset):
    def __init__(self, csv_file, token_dir, max_length=750):
        self.df = pd.read_csv(csv_file)
        self.token_dir = token_dir
        self.max_length = max_length
        
        print(f"\n📊 CoSG Dataset:")
        print(f"   Total: {len(self.df):,}")
        print(f"   Bonafide: {(self.df['label'] == 0).sum():,}")
        print(f"   Spoof: {(self.df['label'] == 1).sum():,}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        token_path = os.path.join(self.token_dir, os.path.splitext(row['filename'])[0] + '.npy')
        
        try:
            tokens = np.load(token_path)
            if tokens.shape[1] > self.max_length:
                tokens = tokens[:, :self.max_length]
            else:
                pad_len = self.max_length - tokens.shape[1]
                tokens = np.pad(tokens, ((0,0), (0,pad_len)), mode='constant')
        except:
            tokens = np.zeros((8, self.max_length), dtype=np.int64)
        
        return torch.LongTensor(tokens), row['label'], row['filename']

# ============ LOAD & EVALUATE ============
print("\n" + "="*80)
print("[1/3] Loading 1M Model")
print("="*80)
model = LSTMClassifier().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()
print("✅ Model loaded")

print("\n" + "="*80)
print("[2/3] Loading CoSG Dataset")
print("="*80)
dataset = CoSGDataset(CSV_FILE, TOKEN_DIR)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

print("\n" + "="*80)
print("[3/3] Evaluating")
print("="*80)
all_labels, all_preds, all_probs, all_files = [], [], [], []

with torch.no_grad():
    for tokens, labels, files in tqdm(loader, desc="Testing"):
        tokens = tokens.to(DEVICE)
        outputs = model(tokens)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_files.extend(files)

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# ============ CALCULATE METRICS ============
acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
cm = confusion_matrix(all_labels, all_preds)
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
fnr = 1 - tpr
eer_idx = np.nanargmin(np.abs(fnr - fpr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

# ============ DISPLAY RESULTS ============
print("\n" + "="*80)
print("📊 COSG CROSS-DATASET EVALUATION RESULTS")
print("="*80)

print(f"\n🎯 PERFORMANCE:")
print(f"   Accuracy: {acc*100:.2f}%")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1-Score: {f1:.4f}")
print(f"   EER: {eer*100:.2f}%")
print(f"   ROC-AUC: {roc_auc:.4f}")

print(f"\n📈 GENERALIZATION GAP:")
print(f"   CodecFake Test:  97.15% (in-domain)")
print(f"   CoSG Test:       {acc*100:.2f}% (cross-domain)")
print(f"   Drop:            {(97.15 - acc*100):.2f}%")

print(f"\n🎭 Confusion Matrix:")
tn, fp, fn, tp = cm.ravel()
print(f"                Predicted")
print(f"            Bonafide  Spoof")
print(f"Actual Bonafide {tn:>5}  {fp:>5}")
print(f"       Spoof    {fn:>5}  {tp:>5}")

print(f"\n📋 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Bonafide', 'Spoof'], digits=4))

# ============ SAVE RESULTS ============
results_df = pd.DataFrame({
    'filename': all_files,
    'true_label': all_labels,
    'predicted_label': all_preds,
    'spoof_probability': all_probs
})
results_df.to_csv(f"{RESULTS_DIR}/cosg_predictions.csv", index=False)

with open(f"{RESULTS_DIR}/cosg_summary.txt", 'w') as f:
    f.write("COSG Cross-Dataset Evaluation\n")
    f.write("="*50 + "\n\n")
    f.write(f"Accuracy: {acc*100:.2f}%\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write(f"EER: {eer*100:.2f}%\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
    f.write(f"Generalization Gap: {(97.15 - acc*100):.2f}%\n")

print(f"\n✅ Results saved to: {RESULTS_DIR}/")
print("="*80)

