# Quick Start Guide - Nepali FastSpeech2

## TL;DR - Fast Track

```bash
# 1. Make setup script executable
chmod +x setup_preprocessing.sh

# 2. Run everything
./setup_preprocessing.sh

# 3. Update config with values from stats.json
# Edit nepali_config_template.yaml

# 4. Start training
python train.py --config nepali_config_template.yaml
```

That's it! The script handles everything automatically.

---

## Detailed Steps

### Prerequisites

You should have:
- ✅ 2064 TextGrid files in `MFA_filelist/`
- ✅ 2064 WAV files in `wavs/`
- ✅ FastSpeech2 repository cloned
- ✅ Python 3.7+ installed
- ✅ CUDA/GPU setup (recommended)

### Step 1: Run Preprocessing (Automated)

```bash
# Make the setup script executable
chmod +x setup_preprocessing.sh

# Run it
./setup_preprocessing.sh
```

This will:
1. Install all dependencies
2. Extract phoneme symbols from TextGrids
3. Process all 2064 audio files
4. Extract durations, pitch, energy, mel-spectrograms
5. Compute normalization statistics
6. Create train/val splits
7. Verify everything

**Expected time**: 10-30 minutes (depends on CPU)

### Step 2: Check Results

After preprocessing, check these files:

```bash
# Check statistics
cat preprocessed_data/nepali/stats.json

# Check symbols count
wc -l preprocessed_data/nepali/symbols.txt

# Check train/val split
wc -l preprocessed_data/nepali/train.txt
wc -l preprocessed_data/nepali/val.txt
```

### Step 3: Update Configuration

Open `nepali_config_template.yaml` and update:

```yaml
# 1. Update symbol count (line ~30)
n_symbols: XX  # Replace with count from symbols.txt

# 2. Update pitch values (from stats.json)
pitch:
  min: XX.X  # pitch.min from stats.json
  max: XX.X  # pitch.max from stats.json

# 3. Update energy values (from stats.json)
energy:
  min: X.X   # energy.min from stats.json
  max: XX.X  # energy.max from stats.json
```

### Step 4: Adjust Training Settings (Optional)

Based on your GPU:

```yaml
# For 8GB GPU
batch_size: 16

# For 12GB GPU
batch_size: 24

# For 16GB+ GPU
batch_size: 32
```

### Step 5: Start Training

```bash
# Start training
python train.py --config nepali_config_template.yaml

# Monitor with TensorBoard
tensorboard --logdir output/log/nepali --port 6006
```

Open browser: http://localhost:6006

### Step 6: Monitor Training

Watch these metrics in TensorBoard:
- **Mel Loss**: Should decrease to < 0.5
- **Duration Loss**: Should decrease to < 2.0
- **Pitch Loss**: Should decrease to < 0.3
- **Energy Loss**: Should decrease to < 0.3
- **Total Loss**: Should decrease to < 1.5

### Step 7: Test Synthesis (After ~20k steps)

```bash
# Synthesize a test sentence
python synthesize.py \
  --text "नमस्ते नेपाल" \
  --restore_step 20000 \
  --config nepali_config_template.yaml
```

---

## Manual Step-by-Step (If Automatic Fails)

### 1. Install Dependencies

```bash
pip install numpy scipy librosa soundfile textgrid pyworld tqdm
```

### 2. Extract Phoneme Symbols

```bash
python extract_phoneme_symbols.py \
    --textgrid_dir MFA_filelist \
    --output preprocessed_data/nepali/symbols.txt \
    --stats
```

### 3. Run Preprocessing

```bash
python nepali_preprocessing.py \
    --textgrid_dir MFA_filelist \
    --wav_dir wavs \
    --output_dir preprocessed_data/nepali \
    --sample_rate 22050 \
    --val_ratio 0.05
```

### 4. Verify Results

```bash
python verify_preprocessing.py --data_dir preprocessed_data/nepali
```

### 5. Check a Specific Sample

```bash
# Get first basename from train.txt
SAMPLE=$(head -1 preprocessed_data/nepali/train.txt | cut -d'|' -f1)

# Verify it
python verify_preprocessing.py \
    --data_dir preprocessed_data/nepali \
    --sample $SAMPLE
```

---

## Training Timeline (100k iterations)

On a single GPU:

| Steps | Time | What to Expect |
|-------|------|----------------|
| 0-1k | 30 min | Random noise, high loss |
| 1k-5k | 2 hours | Basic mel structure forming |
| 5k-20k | 8 hours | Recognizable speech patterns |
| 20k-50k | 20 hours | Clear speech, some artifacts |
| 50k-80k | 32 hours | High quality speech |
| 80k-100k | 40 hours | Polished, natural speech |

**Total**: ~40-50 hours on RTX 3090

---

## Common Issues

### Issue 1: CUDA out of memory
```bash
# Reduce batch size in config
batch_size: 8  # or even 4
```

### Issue 2: Import errors
```bash
# Install missing package
pip install <package_name>
```

### Issue 3: TextGrid parsing errors
```bash
# Check TextGrid format
python -c "import textgrid; tg = textgrid.TextGrid.fromFile('MFA_filelist/file1.TextGrid'); print(tg)"
```

### Issue 4: Duration mismatch warnings
This is usually fine if < 5 frames difference. The script auto-adjusts.

### Issue 5: Training loss not decreasing
- Check learning rate (try 0.0001)
- Check if data is normalized correctly
- Verify batch size is appropriate
- Check for NaN values in preprocessed data

---

## Quick Commands Reference

```bash
# Full preprocessing pipeline
./setup_preprocessing.sh

# Extract symbols only
python extract_phoneme_symbols.py --textgrid_dir MFA_filelist

# Preprocess only
python nepali_preprocessing.py --wav_dir wavs --textgrid_dir MFA_filelist

# Verify preprocessing
python verify_preprocessing.py

# Check specific sample
python verify_preprocessing.py --sample <basename>

# Start training
python train.py --config nepali_config_template.yaml

# Resume training
python train.py --config nepali_config_template.yaml --checkpoint output/ckpt/nepali/checkpoint_20000.pth

# TensorBoard
tensorboard --logdir output/log/nepali

# Synthesize
python synthesize.py --text "test" --restore_step 50000
```

---

## File Structure After Preprocessing

```
preprocessed_data/nepali/
├── duration/           # 2064 .npy files (phoneme durations)
├── pitch/              # 2064 .npy files (F0 contours)
├── energy/             # 2064 .npy files (energy values)
├── mel/                # 2064 .npy files (mel-spectrograms)
├── train.txt           # ~1960 training samples
├── val.txt             # ~104 validation samples
├── metadata.json       # Full metadata
├── stats.json          # Normalization statistics
├── symbols.txt         # Phoneme symbols (text)
├── symbols.py          # Phoneme symbols (Python)
└── failed_files.txt    # Failed files (if any)
```

---

## Expected Output Sizes

- **Duration files**: ~5-10 KB each
- **Pitch files**: ~50-100 KB each
- **Energy files**: ~50-100 KB each
- **Mel files**: ~200-300 KB each
- **Total**: ~700-800 MB for 2064 files

---

## Next Steps After Training

1. **Evaluate model**: Test on held-out sentences
2. **Train vocoder**: HiFi-GAN or MelGAN for Nepali
3. **Fine-tune**: Adjust hyperparameters if needed
4. **Deploy**: Create inference API

---

## Getting Help

If you encounter issues:

1. Check `failed_files.txt` for problematic files
2. Run verification: `python verify_preprocessing.py`
3. Check logs in `output/log/nepali/`
4. Verify TextGrid format matches expectations

Good luck with your Nepali TTS model! 🎉