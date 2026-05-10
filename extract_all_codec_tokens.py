#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

print("=" * 80)
print("SMART TOKEN EXTRACTION (Extract Once, Use Many)")
print("=" * 80)

DATASETS_DIR  = "/Data1/cse_24203109/datasets_by_codec"
PROCESSED_DIR = "/Data1/cse_24203109/processed_data"
AUDIO_DIR     = "/Data1/cse_24203109/codecfake_dataset_new/Codecfake_dataset_CoRS"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device.upper()}")

print("\nLoading EnCodec...")
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model = model.to(device)
model.eval()
print("Loaded!")

print("\nScanning all codec CSVs...")
all_files      = set()
codec_to_label = {}

codec_csvs = sorted([f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')])

for csv_file in codec_csvs:
    codec_name = csv_file.replace('.csv', '')
    csv_path   = os.path.join(DATASETS_DIR, csv_file)
    df         = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        filename = row['filename']
        label    = row['label']
        all_files.add(filename)

        if filename not in codec_to_label:
            codec_to_label[filename] = label

print(f"Found {len(all_files):,} unique files")

print(f"\nExtracting {len(all_files):,} unique files...")

bonafide_files = {f: codec_to_label[f] for f in all_files if codec_to_label[f] == 'bonafide'}
spoof_files    = {f: codec_to_label[f] for f in all_files if codec_to_label[f] != 'bonafide'}

print(f"   Bonafide: {len(bonafide_files):,}")
print(f"   Spoof: {len(spoof_files):,}")

bonafide_dir = os.path.join(PROCESSED_DIR, "tokens_bonafide_combined")
os.makedirs(bonafide_dir, exist_ok=True)

print(f"\n[1/2] Extracting Bonafide tokens...")
success = 0
for filename in tqdm(bonafide_files.keys(), desc="Bonafide"):
    audio_path = os.path.join(AUDIO_DIR, filename)
    token_path = os.path.join(bonafide_dir, f"{os.path.splitext(filename)[0]}.npy")

    if os.path.exists(token_path):
        success += 1
        continue

    try:
        wav, sr = torchaudio.load(audio_path)
        wav     = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav     = wav.unsqueeze(0).to(device)

        max_samples = int(10.0 * model.sample_rate)
        if wav.shape[2] > max_samples:
            wav = wav[:, :, :max_samples]

        with torch.no_grad():
            encoded_frames = model.encode(wav)
            codes  = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
            tokens = codes.cpu().numpy().squeeze(0)

        np.save(token_path, tokens)
        success += 1
    except:
        pass

print(f"Bonafide: {success:,}/{len(bonafide_files):,}")

print(f"\n[2/2] Extracting Spoof tokens (by codec)...")

for csv_file in codec_csvs:
    codec_name = csv_file.replace('.csv', '')
    csv_path   = os.path.join(DATASETS_DIR, csv_file)
    df         = pd.read_csv(csv_path)

    spoof_df = df[df['label'] != 'bonafide']
    if len(spoof_df) == 0:
        continue

    spoof_dir = os.path.join(PROCESSED_DIR, f"tokens_spoof_{codec_name}")
    os.makedirs(spoof_dir, exist_ok=True)

    success = 0
    for _, row in tqdm(spoof_df.iterrows(), desc=codec_name, leave=False):
        audio_path = os.path.join(AUDIO_DIR, row['filename'])
        token_path = os.path.join(spoof_dir, f"{os.path.splitext(row['filename'])[0]}.npy")

        if os.path.exists(token_path):
            success += 1
            continue

        try:
            wav, sr = torchaudio.load(audio_path)
            wav     = convert_audio(wav, sr, model.sample_rate, model.channels)
            wav     = wav.unsqueeze(0).to(device)

            max_samples = int(10.0 * model.sample_rate)
            if wav.shape[2] > max_samples:
                wav = wav[:, :, :max_samples]

            with torch.no_grad():
                encoded_frames = model.encode(wav)
                codes  = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
                tokens = codes.cpu().numpy().squeeze(0)

            np.save(token_path, tokens)
            success += 1
        except:
            pass

    print(f"   {codec_name}: {success:,}/{len(spoof_df):,}")

print(f"\n" + "=" * 80)
print("SMART EXTRACTION COMPLETE!")
print("=" * 80)
print(f"\nStructure created:")
print(f"   tokens_bonafide_combined/  (44k files)")
print(f"   tokens_spoof_DAC24/")
print(f"   tokens_spoof_Encodec_6b24k/")
print(f"   ... (29 codec spoof dirs)")
print("=" * 80)
