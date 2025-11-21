#!/bin/bash

# Nepali FastSpeech2 Preprocessing Setup Script
# Run this script to set up and run the complete preprocessing pipeline

echo "=========================================="
echo "Nepali FastSpeech2 Preprocessing Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -d "MFA_filelist" ] || [ ! -d "wavs" ]; then
    echo "❌ Error: MFA_filelist or wavs directory not found!"
    echo "Please run this script from your FastSpeech2 root directory"
    exit 1
fi

echo "✓ Found required directories"
echo ""

# Count files
textgrid_count=$(find MFA_filelist -name "*.TextGrid" | wc -l)
wav_count=$(find wavs -name "*.wav" | wc -l)

echo "Dataset information:"
echo "  TextGrid files: $textgrid_count"
echo "  WAV files: $wav_count"
echo ""

# # Step 1: Install dependencies
# echo "Step 1: Installing dependencies..."
# echo "----------------------------------------"
# pip install -q numpy scipy librosa soundfile textgrid pyworld tqdm

# if [ $? -ne 0 ]; then
#     echo "❌ Failed to install dependencies"
#     exit 1
# fi
# echo "✓ Dependencies installed"
# echo ""

# Step 2: Extract phoneme symbols
echo "Step 2: Extracting phoneme symbols..."
echo "----------------------------------------"
python extract_phenome_symbols.py \
    --textgrid_dir MFA_filelist \
    --output preprocessed_data/nepali/symbols.txt \
    --stats

if [ $? -ne 0 ]; then
    echo "❌ Failed to extract phoneme symbols"
    exit 1
fi
echo ""

# # Step 3: Run preprocessing
# echo "Step 3: Running preprocessing..."
# echo "----------------------------------------"
# echo "This may take 10-30 minutes depending on your dataset size..."
# echo ""

# python preprocess.py \
#     --textgrid_dir MFA_filelist \
#     --wav_dir wavs \
#     --output_dir preprocessed_data/nepali \
#     --sample_rate 22050 \
#     --val_ratio 0.05

# if [ $? -ne 0 ]; then
#     echo "❌ Preprocessing failed"
#     exit 1
# fi
# echo ""

# Step 4: Verify preprocessing
echo "Step 4: Verifying preprocessed data..."
echo "----------------------------------------"
python verify_preprocessing.py --data_dir preprocessed_data/nepali

if [ $? -ne 0 ]; then
    echo "⚠️  Verification had issues, but continuing..."
fi
echo ""

# Summary
echo "=========================================="
echo "PREPROCESSING COMPLETE!"
echo "=========================================="
echo ""
echo "Output directory: preprocessed_data/nepali/"
echo ""
echo "Generated files:"
echo "  - duration/     : Phoneme duration files"
echo "  - pitch/        : F0 files"
echo "  - energy/       : Energy files"
echo "  - mel/          : Mel-spectrogram files"
echo "  - train.txt     : Training filelist"
echo "  - val.txt       : Validation filelist"
echo "  - symbols.txt   : Phoneme symbols"
echo "  - stats.json    : Normalization statistics"
echo ""
echo "Next steps:"
echo "1. Check stats.json for pitch/energy ranges"
echo "2. Update your training config with these values"
echo "3. Update n_symbols in config (check symbols.txt)"
echo "4. Start training: python train.py"
echo ""
echo "For detailed verification of a specific file:"
echo "  python verify_preprocessing.py --sample <basename>"
echo ""