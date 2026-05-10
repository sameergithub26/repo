#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import random
import sys

sys.path.insert(0, '/Data1/cse_24203109/scripts')
from model_token_lstm_se import TokenLSTM_SE

print("=" * 80)
print("RETRAINING TOP-4 MODELS WITH SE-ATTENTION")
print("=" * 80)

PROCESSED_DIR = "/Data1/cse_24203109/processed_data"
MODELS_DIR    = "/Data1/cse_24203109/models/ensemble_4_with_se"
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {DEVICE}")


class CodecDataset(Dataset):
    def __init__(self, codec_name, bonafide_dir, spoof_dir, split='train', max_length=750):
        self.max_length = max_length
        self.samples    = []

        bonafide_files = sorted([f for f in os.listdir(bonafide_dir) if f.endswith('.npy')])
        spoof_files    = sorted([f for f in os.listdir(spoof_dir) if f.endswith('.npy')])

        total_bonafide = len(bonafide_files)
        total_spoof    = len(spoof_files)

        train_bonafide = int(0.8 * total_bonafide)
        train_spoof    = int(0.8 * total_spoof)
        val_bonafide   = int(0.1 * total_bonafide)
        val_spoof      = int(0.1 * total_spoof)

        if split == 'train':
            for f in bonafide_files[:train_bonafide]:
                self.samples.append((os.path.join(bonafide_dir, f), 0))
            for f in spoof_files[:train_spoof]:
                self.samples.append((os.path.join(spoof_dir, f), 1))
        elif split == 'val':
            for f in bonafide_files[train_bonafide:train_bonafide + val_bonafide]:
                self.samples.append((os.path.join(bonafide_dir, f), 0))
            for f in spoof_files[train_spoof:train_spoof + val_spoof]:
                self.samples.append((os.path.join(spoof_dir, f), 1))
        elif split == 'test':
            for f in bonafide_files[train_bonafide + val_bonafide:]:
                self.samples.append((os.path.join(bonafide_dir, f), 0))
            for f in spoof_files[train_spoof + val_spoof:]:
                self.samples.append((os.path.join(spoof_dir, f), 1))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_path, label = self.samples[idx]
        try:
            tokens = np.load(token_path)
            if tokens.shape[1] > self.max_length:
                tokens = tokens[:, :self.max_length]
            else:
                pad_len = self.max_length - tokens.shape[1]
                tokens  = np.pad(tokens, ((0, 0), (0, pad_len)), mode='constant')
            return torch.LongTensor(tokens), label
        except:
            return torch.zeros((8, self.max_length), dtype=torch.long), 0


codecs_to_retrain = [
    'HiFi_Codec_16k_320d_large_universal',
    'bigcodec',
    'FunCodec_en-libritts-16k-gr1nq32ds320',
    'FACodec_encodec-decoder-v2_16k',
]

print(f"\nTraining 4 models with SE-Attention:")
for i, codec in enumerate(codecs_to_retrain, 1):
    codec_type = "TIME" if i <= 3 else "FREQ"
    print(f"   {i}. {codec} ({codec_type})")

bonafide_dir   = os.path.join(PROCESSED_DIR, "tokens_bonafide_combined")
trained_models = []
failed_codecs  = []

for idx, codec_name in enumerate(codecs_to_retrain, 1):
    spoof_dir = os.path.join(PROCESSED_DIR, f"tokens_spoof_{codec_name}")

    print(f"\n{'=' * 80}")
    print(f"[{idx}/{len(codecs_to_retrain)}] {codec_name}")
    print('=' * 80)

    try:
        train_dataset = CodecDataset(codec_name, bonafide_dir, spoof_dir, split='train')
        val_dataset   = CodecDataset(codec_name, bonafide_dir, spoof_dir, split='val')
        test_dataset  = CodecDataset(codec_name, bonafide_dir, spoof_dir, split='test')

        print(f"   Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4)
        val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4)
        test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=4)

        model = TokenLSTM_SE().to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_val_acc     = 0
        best_model_state = None
        patience         = 0

        for epoch in range(15):
            model.train()
            train_preds  = []
            train_labels = []

            for tokens, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(tokens)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            train_acc = accuracy_score(train_labels, train_preds)

            model.eval()
            val_preds  = []
            val_labels = []
            with torch.no_grad():
                for tokens, labels in val_loader:
                    tokens  = tokens.to(DEVICE)
                    outputs = model(tokens)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_acc = accuracy_score(val_labels, val_preds)
            print(f"   Epoch {epoch+1:2d} | Train: {train_acc:.4f} | Val: {val_acc:.4f}", end="")

            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                best_model_state = model.state_dict().copy()
                patience         = 0
                print(" *")
            else:
                patience += 1
                print()
                if patience >= 3:
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)
            model.eval()
            test_preds  = []
            test_labels = []

            with torch.no_grad():
                for tokens, labels in test_loader:
                    tokens  = tokens.to(DEVICE)
                    outputs = model(tokens)
                    _, preds = torch.max(outputs, 1)
                    test_preds.extend(preds.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())

            test_acc   = accuracy_score(test_labels, test_preds)
            model_path = os.path.join(MODELS_DIR, f"{codec_name}.pth")
            torch.save(best_model_state, model_path)
            trained_models.append((codec_name, best_val_acc, test_acc))
            print(f"\n   Val: {best_val_acc:.4f} | Test: {test_acc:.4f}")

    except Exception as e:
        print(f"\n   ERROR: {str(e)[:100]}")
        failed_codecs.append(codec_name)

print(f"\n{'=' * 80}")
print(f"TRAINING WITH SE-ATTENTION COMPLETE!")
print(f"{'=' * 80}")
print(f"\nSuccessfully trained: {len(trained_models)}/4\n")

if trained_models:
    print(f"Model Results:")
    print(f"{'Codec':<50} {'Val':<10} {'Test':<10}")
    print("-" * 70)
    for codec, val_acc, test_acc in trained_models:
        print(f"{codec:<50} {val_acc:.4f}      {test_acc:.4f}")

print(f"\nModels saved to: {MODELS_DIR}")
print("=" * 80)
