#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from pathlib import Path

DATASET_DIR = "/Data1/cse_24203109/codecfake_dataset/Codecfake_plus_CoRS"
OUTPUT_DIR = "/Data1/cse_24203109/processed_data"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "balanced_1M_dataset_new.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("CREATING BALANCED 1M DATASET (CORRECT - WITH REAL AUDIO AUGMENTATION)")
print("="*80)

# ============ SCAN ACTUAL FILES ============
print("\n[1/3] Scanning audio files...")
import re

wav_files = list(Path(DATASET_DIR).glob("*.wav"))
print(f"Total files found: {len(wav_files):,}")

bonafide_pattern = re.compile(r'^[ps]\d+_\d+\.wav$')
spoof_pattern = re.compile(r'^[ps]\d+_\d+_.*\.wav$')

bonafide_files = []
spoof_files = {}  # Group by codec

for wav_file in wav_files:
    filename = wav_file.name
    
    if bonafide_pattern.match(filename):
        bonafide_files.append(filename)
    elif spoof_pattern.match(filename):
        # Extract codec name
        match = re.match(r'^[ps]\d+_\d+_(.*?)\.wav$', filename)
        if match:
            codec = match.group(1)
            if codec not in spoof_files:
                spoof_files[codec] = []
            spoof_files[codec].append(filename)

print(f"✓ Bonafide files: {len(bonafide_files):,}")
print(f"✓ Spoof files: {sum(len(v) for v in spoof_files.values()):,}")
print(f"✓ Unique codecs: {len(spoof_files)}")

# ============ BONAFIDE STRATEGY (WITH REAL AUGMENTATION) ============
print("\n[2/3] Creating bonafide dataset with REAL audio augmentation metadata...")
print("="*80)
print("BONAFIDE: ALL ORIGINALS + AUGMENTATIONS (REAL AUDIO)")
print("="*80)

TARGET_BONAFIDE = 500000
current_bonafide = len(bonafide_files)
need_augmented = TARGET_BONAFIDE - current_bonafide

print(f"\nOriginal bonafide: {current_bonafide:,}")
print(f"Target bonafide: {TARGET_BONAFIDE:,}")
print(f"Augmentations needed: {need_augmented:,}")

# Augmentation types (10 types)
aug_types = [
    'time_stretch_0.9',
    'time_stretch_1.1',
    'pitch_shift_-2',
    'pitch_shift_+2',
    'noise_snr20',
    'noise_snr15',
    'reverb_room',
    'compression_mp3',
    'speed_perturb_0.95',
    'speed_perturb_1.05'
]

bonafide_dataset = []

# 1. Add all originals
print("\nAdding original bonafide files...")
for filename in bonafide_files:
    bonafide_dataset.append({
        'filename': filename,
        'label': 0,
        'codec': 'original',
        'augmentation': 'none',
        'is_augmented': False
    })

# 2. Calculate how many augmentations per file
augs_per_file = need_augmented // current_bonafide
remaining = need_augmented % current_bonafide

print(f"\nAugmentation plan:")
print(f" Each of {current_bonafide:,} files → {augs_per_file} augmentations")
print(f" Plus {remaining:,} additional random augmentations")

# 3. Create augmentations (systematic)
print("\nCreating augmented bonafide entries...")
for aug_idx in range(augs_per_file):
    aug_type = aug_types[aug_idx % len(aug_types)]
    for filename in bonafide_files:
        bonafide_dataset.append({
            'filename': filename,
            'label': 0,
            'codec': 'original',
            'augmentation': f'{aug_type}_v{aug_idx}',
            'is_augmented': True
        })

# 4. Add remaining augmentations
np.random.seed(42)
remaining_samples = np.random.choice(bonafide_files, size=remaining, replace=False)
for filename in remaining_samples:
    aug_type = np.random.choice(aug_types)
    bonafide_dataset.append({
        'filename': filename,
        'label': 0,
        'codec': 'original',
        'augmentation': f'{aug_type}_extra',
        'is_augmented': True
    })

bonafide_df = pd.DataFrame(bonafide_dataset)
print(f"\n✓ Total bonafide samples: {len(bonafide_df):,}")
print(f"  Original: {(~bonafide_df['is_augmented']).sum():,}")
print(f"  Augmented: {bonafide_df['is_augmented'].sum():,}")

# ============ SPOOF STRATEGY ============
print("\n" + "="*80)
print("[3/3] Creating spoof dataset...")
print("="*80)
print("SPOOF: BALANCED SAMPLING ACROSS CODECS")
print("="*80)

TARGET_SPOOF = 500000
total_spoof = sum(len(v) for v in spoof_files.values())

print(f"\nTotal spoof files: {total_spoof:,}")
print(f"Target: {TARGET_SPOOF:,}")

# Sample proportionally from each codec
spoof_dataset = []
samples_per_codec = {}

for codec, files in spoof_files.items():
    # Calculate proportional sample size
    proportion = len(files) / total_spoof
    n_samples = int(TARGET_SPOOF * proportion)
    
    # Ensure we don't sample more than available
    n_samples = min(n_samples, len(files))
    
    # Random sample
    sampled = np.random.choice(files, size=n_samples, replace=False)
    samples_per_codec[codec] = len(sampled)
    
    for filename in sampled:
        spoof_dataset.append({
            'filename': filename,
            'label': 1,
            'codec': codec,
            'augmentation': 'none',
            'is_augmented': False
        })

# Adjust to exactly 500k if needed
current_spoof = len(spoof_dataset)
if current_spoof < TARGET_SPOOF:
    shortfall = TARGET_SPOOF - current_spoof
    print(f"\nShortfall: {shortfall:,} - adding more samples...")
    
    # Add more from largest codecs
    all_remaining = []
    for codec, files in spoof_files.items():
        sampled_set = set([d['filename'] for d in spoof_dataset if d['codec'] == codec])
        remaining = [f for f in files if f not in sampled_set]
        all_remaining.extend([(f, codec) for f in remaining])
    
    additional = np.random.choice(len(all_remaining), size=shortfall, replace=False)
    for idx in additional:
        filename, codec = all_remaining[idx]
        spoof_dataset.append({
            'filename': filename,
            'label': 1,
            'codec': codec,
            'augmentation': 'none',
            'is_augmented': False
        })

spoof_df = pd.DataFrame(spoof_dataset)
print(f"\n✓ Total spoof samples: {len(spoof_df):,}")

print(f"\nCodec distribution (top 15):")
codec_dist = spoof_df['codec'].value_counts()
for codec, count in codec_dist.head(15).items():
    print(f"  {codec}: {count:,} ({count/len(spoof_df)*100:.1f}%)")

# ============ COMBINE AND SAVE ============
print("\n" + "="*80)
print("FINAL BALANCED DATASET")
print("="*80)

balanced_df = pd.concat([bonafide_df, spoof_df], ignore_index=True)

# Shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal dataset statistics:")
print(f"  Total samples: {len(balanced_df):,}")
print(f"  Bonafide (label=0): {(balanced_df['label'] == 0).sum():,}")
print(f"  Spoof (label=1): {(balanced_df['label'] == 1).sum():,}")
print(f"  Balance ratio: {(balanced_df['label']==0).sum() / (balanced_df['label']==1).sum():.2f}:1")
print(f"\n  Original files: {(~balanced_df['is_augmented']).sum():,}")
print(f"  Augmented files: {balanced_df['is_augmented'].sum():,}")

# Save
balanced_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Saved to: {OUTPUT_CSV}")

# Show first few rows
print(f"\nFirst 10 samples:")
print(balanced_df.head(10)[['filename', 'label', 'codec', 'augmentation', 'is_augmented']])

print("\n" + "="*80)
print("✅ SUCCESS! NEW 1M DATASET CREATED!")
print("="*80)
print("\nKey differences from old approach:")
print(" ✓ Augmentation metadata points to REAL audio files")
print(" ✓ Ready for REAL audio augmentation during token extraction")
print(" ✓ No fake token-level modifications needed")
print("\nNext steps:")
print(" 1. Run: python apply_audio_augmentation_and_extract.py")
print(" 2. This will apply REAL audio augmentation + extract tokens")
print(" 3. Create splits and train model")
print("\n" + "="*80)
