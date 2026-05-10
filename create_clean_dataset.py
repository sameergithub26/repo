#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import re

print("=" * 80)
print("CREATING CLEAN CODECFAKE DATASET")
print("=" * 80)

OLD_CORS_DIR        = "/Data1/cse_24203109/codecfake_dataset/Codecfake_plus_CoRS"
OLD_LABELS_FILE     = "/Data1/cse_24203109/codecfake_dataset/CoRS_labels.txt"
INTERSECTION_FILE   = "/Data1/cse_24203109/intersection_files.txt"
INTERSECTION_LABELS = "/Data1/cse_24203109/intersection_labels.txt"

NEW_BASE_DIR   = "/Data1/cse_24203109/codecfake_dataset_new"
NEW_CORS_DIR   = f"{NEW_BASE_DIR}/Codecfake_dataset_CoRS"
NEW_LABELS_FILE = f"{NEW_BASE_DIR}/CoRS_labels.txt"

print("\n[1] CREATING DIRECTORIES")
print("-" * 80)

os.makedirs(NEW_CORS_DIR, exist_ok=True)
print(f"Created: {NEW_BASE_DIR}")
print(f"Created: {NEW_CORS_DIR}")

print("\n[2] READING INTERSECTION FILES")
print("-" * 80)

if not os.path.exists(INTERSECTION_FILE):
    print(f"intersection_files.txt not found!")
    print(f"Run audit_full.py first to create it")
    exit(1)

if not os.path.exists(INTERSECTION_LABELS):
    print(f"intersection_labels.txt not found!")
    print(f"Run audit_full.py first to create it")
    exit(1)

with open(INTERSECTION_FILE, 'r') as f:
    intersection_files = [line.strip() for line in f if line.strip()]

print(f"Found {len(intersection_files):,} files to copy")

with open(INTERSECTION_LABELS, 'r') as f:
    intersection_labels = [line.strip() for line in f if line.strip()]

print(f"Found {len(intersection_labels):,} labels to copy")

print("\n[3] COPYING WAV FILES")
print("-" * 80)

copied = 0
failed = 0

for i, filename in enumerate(intersection_files):
    src = os.path.join(OLD_CORS_DIR, filename)
    dst = os.path.join(NEW_CORS_DIR, filename)

    try:
        shutil.copy2(src, dst)
        copied += 1

        if (i + 1) % 50000 == 0:
            print(f"  Copied {copied:,} / {len(intersection_files):,}...")

    except Exception as e:
        print(f"  Failed to copy {filename}: {e}")
        failed += 1

print(f"\nCopied: {copied:,} files")
if failed > 0:
    print(f"Failed: {failed:,} files")

print("\n[4] WRITING LABELS FILE")
print("-" * 80)

with open(NEW_LABELS_FILE, 'w') as f:
    for line in intersection_labels:
        f.write(f"{line}\n")

print(f"Written labels to: {NEW_LABELS_FILE}")
print(f"  Total labels: {len(intersection_labels):,}")

print("\n[5] VERIFICATION")
print("-" * 80)

new_files = list(Path(NEW_CORS_DIR).glob("*.wav"))
print(f"\nFiles in new directory: {len(new_files):,}")

with open(NEW_LABELS_FILE, 'r') as f:
    new_labels = [line.strip() for line in f if line.strip()]

print(f"Labels in new file: {len(new_labels):,}")

bonafide_count = 0
spoof_count    = 0
codecs         = {}

for line in new_labels:
    parts = line.split()
    if len(parts) >= 3:
        label    = parts[2]
        filename = parts[1]

        if label == 'bonafide':
            bonafide_count += 1
        elif label == 'spoof':
            match = re.match(r'^[ps]\d+_\d+_(.+)\.wav$', filename)
            if match:
                codec = match.group(1)
                if codec not in codecs:
                    codecs[codec] = 0
                codecs[codec] += 1
            spoof_count += 1

print(f"\nNEW DATASET BREAKDOWN:")
print(f"  Bonafide: {bonafide_count:,}")
print(f"  Spoof: {spoof_count:,}")
print(f"  Unique codecs: {len(codecs)}")

print(f"\nCODECS IN NEW DATASET ({len(codecs)}):")
for codec in sorted(codecs.keys()):
    count = codecs[codec]
    print(f"  {codec:50s}: {count:>6,}")

print(f"\n[6] FINAL VERIFICATION")
print("-" * 80)

missing = 0
for fname in intersection_files:
    if not os.path.exists(os.path.join(NEW_CORS_DIR, fname)):
        missing += 1

if missing == 0:
    print(f"\nSUCCESS!")
    print(f"   Clean dataset created at: {NEW_BASE_DIR}")
    print(f"\n   Structure:")
    print(f"   {NEW_BASE_DIR}/")
    print(f"   |-- Codecfake_dataset_CoRS/  ({len(new_files):,} WAV files)")
    print(f"   |-- CoRS_labels.txt  ({len(new_labels):,} labels)")
    print(f"\n   Ready to use for training!")
else:
    print(f"\n{missing:,} files missing after copy!")

print("\n" + "=" * 80)
