# Nepali FastSpeech2 Preprocessing Guide

## Step-by-Step Instructions

### Step 1: Install Dependencies

```bash
# Install required packages
pip install numpy scipy librosa soundfile textgrid pyworld tqdm
```

### Step 2: Verify Your Directory Structure

Make sure you have:
```
FastSpeech2/
├── MFA_filelist/          # Your TextGrid files
│   ├── file1.TextGrid
│   ├── file2.TextGrid
│   └── ...
├── wavs/                  # Your audio files
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
├── text/                  # Your transcriptions (if needed)
└── nepali_mfa_dict.txt   # Your phoneme dictionary
```

### Step 3: Extract Phoneme Symbols

First, extract all unique phonemes from your TextGrid files:

```bash
python extract_phoneme_symbols.py --textgrid_dir MFA_filelist --output preprocessed_data/nepali/symbols.txt --stats
```

This will:
- Scan all TextGrid files
- Extract unique phonemes
- Create `symbols.txt` and `symbols.py`
- Print statistics about phoneme distribution

### Step 4: Run Preprocessing

Now preprocess all your data:

```bash
python nepali_preprocessing.py \
    --textgrid_dir MFA_filelist \
    --wav_dir wavs \
    --output_dir preprocessed_data/nepali \
    --sample_rate 22050 \
    --val_ratio 0.05
```

This will:
1. Extract durations from TextGrid files
2. Extract mel-spectrograms from audio
3. Extract pitch (F0) using PyWorld
4. Extract energy from mel-spectrograms
5. Compute normalization statistics
6. Create train/val filelists

### Step 5: Check the Output

After preprocessing, you should have:

```
preprocessed_data/nepali/
├── duration/              # Duration files (.npy)
├── pitch/                 # Pitch files (.npy)
├── energy/                # Energy files (.npy)
├── mel/                   # Mel-spectrogram files (.npy)
├── train.txt              # Training filelist
├── val.txt                # Validation filelist
├── metadata.json          # Metadata for all files
├── stats.json             # Statistics for normalization
├── symbols.txt            # Phoneme symbols (text format)
├── symbols.py             # Phoneme symbols (Python format)
└── failed_files.txt       # List of failed files (if any)
```

### Step 6: Update Configuration

Open `stats.json` and note the values:

```json
{
  "pitch": {
    "min": 80.5,
    "max": 450.2,
    "mean": 180.3,
    "std": 45.6
  },
  "energy": {
    "min": 0.01,
    "max": 85.3,
    "mean": 25.4,
    "std": 12.8
  }
}
```

You'll need these values for your training configuration.

### Step 7: Verify Preprocessing

Quick verification script:

```python
import numpy as np
import json

# Load metadata
with open("preprocessed_data/nepali/metadata.json") as f:
    metadata = json.load(f)

# Check a sample
sample = metadata[0]
basename = sample["basename"]

# Load preprocessed files
duration = np.load(f"preprocessed_data/nepali/duration/{basename}.npy")
pitch = np.load(f"preprocessed_data/nepali/pitch/{basename}.npy")
energy = np.load(f"preprocessed_data/nepali/energy/{basename}.npy")
mel = np.load(f"preprocessed_data/nepali/mel/{basename}.npy")

print(f"Sample: {basename}")
print(f"Phonemes: {sample['phonemes']}")
print(f"Duration shape: {duration.shape}")
print(f"Pitch shape: {pitch.shape}")
print(f"Energy shape: {energy.shape}")
print(f"Mel shape: {mel.shape}")
print(f"Duration sum: {sum(duration)} (should match mel length: {len(mel)})")
```

### Step 8: Create Training Config

Based on your preprocessing, update your training config with:

```yaml
# Audio settings
sampling_rate: 22050
hop_length: 256
n_fft: 1024
win_length: 1024
n_mels: 80

# Data paths
dataset: "nepali"
data_path: "preprocessed_data/nepali"
train_filelist: "preprocessed_data/nepali/train.txt"
val_filelist: "preprocessed_data/nepali/val.txt"

# Normalization (from stats.json)
pitch_min: 80.5
pitch_max: 450.2
energy_min: 0.01
energy_max: 85.3

# Phoneme info
n_symbols: [check symbols.txt for total count]
```

## Common Issues & Solutions

### Issue 1: TextGrid parsing errors
**Solution**: Check if all TextGrid files have a "phones" tier. The script expects tier name exactly as "phones".

### Issue 2: Duration mismatch
**Solution**: The script automatically adjusts small mismatches. If you see many warnings, check your MFA alignment quality.

### Issue 3: Memory issues
**Solution**: If processing 2064 files causes memory issues, you can process in batches by modifying the script.

### Issue 4: Different sample rates
**Solution**: The script automatically resamples audio to match the target sample rate (default 22050 Hz).

## Next Steps

After successful preprocessing:

1. **Update model config**: Use the phoneme count from `symbols.txt`
2. **Verify statistics**: Check if pitch/energy ranges make sense
3. **Start training**: `python train.py --config your_config.yaml`
4. **Monitor training**: Use TensorBoard to track progress

## Training to 100k Iterations

For stable training:
- Use Adam optimizer with learning rate ~0.0001
- Use learning rate warmup (4000 steps)
- Batch size: 16-32 (depending on GPU)
- Gradient clipping: 1.0
- Save checkpoints every 5000 steps
- Expected training time: 6-12 hours on single GPU

## File Sizes (Approximate)

For 2064 files:
- Duration: ~10 MB
- Pitch: ~100 MB
- Energy: ~100 MB
- Mel: ~500 MB
- Total: ~710 MB

Make sure you have enough disk space!