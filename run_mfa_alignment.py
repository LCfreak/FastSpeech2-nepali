import subprocess
import os
import shutil

# Paths (adjust base_path if needed)
base_path = os.path.expanduser('~/Downloads/FastSpeech2')
mfa_input = f'{base_path}/mfa_input'
mfa_output = f'{base_path}/MFA_filelist'
nepali_dict = f'{base_path}/nepali_mfa_dict.txt'
output_model = f'{base_path}/nepali_acoustic_model.zip'

# Clean output directory
if os.path.exists(mfa_output):
    shutil.rmtree(mfa_output)
os.makedirs(mfa_output, exist_ok=True)

# Remove old model
if os.path.exists(output_model):
    os.remove(output_model)

print("="*60)
print("STARTING MFA TRAINING (M1 Mac)")
print("="*60)
print(f"Input: {mfa_input}")
print(f"Dictionary: {nepali_dict}")
print(f"Model output: {output_model}")
print(f"TextGrid output: {mfa_output}")
print(f"\nEstimated time: 1.5-2.5 hours (2000 files)")
print("Output will appear below (real-time logs)")
print("="*60 + "\n")

# Verify input files exist
if not os.path.exists(mfa_input):
    print(f"✗ ERROR: mfa_input directory not found at {mfa_input}")
    print("Please check your paths!")
    exit(1)

if not os.path.exists(nepali_dict):
    print(f"✗ ERROR: Dictionary not found at {nepali_dict}")
    print("Please check your paths!")
    exit(1)

# Count files
wav_files = [f for f in os.listdir(mfa_input) if f.endswith('.wav')]
lab_files = [f for f in os.listdir(mfa_input) if f.endswith('.lab')]

print(f"Found {len(wav_files)} WAV files and {len(lab_files)} LAB files")

if len(wav_files) != len(lab_files):
    print("⚠️  WARNING: WAV and LAB file counts don't match!")
    print(f"WAV: {len(wav_files)}, LAB: {len(lab_files)}")

print()

# MFA training command (optimized for M1 Mac)
cmd = [
    'mfa', 'train',
    mfa_input,
    nepali_dict,
    output_model,
    '--output_directory', mfa_output,
    '--clean',
    '--num_jobs', '6',  # M1 Pro has 8 cores, use 6
    '--verbose'
]

print(f"Command: {' '.join(cmd)}\n")
print("="*60)
print("Training started at:", subprocess.check_output(['date']).decode().strip())
print("="*60 + "\n")

try:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Print real-time output
    for line in process.stdout:
        print(line, end='', flush=True)
    
    process.wait()
    
    if process.returncode == 0:
        print("\n" + "="*60)
        print("✓ MFA TRAINING COMPLETE!")
        print("="*60)
        
        # Verify TextGrids
        textgrids = [f for f in os.listdir(mfa_output) if f.endswith('.TextGrid')]
        
        print(f"\n✓ Generated {len(textgrids)}/{len(lab_files)} TextGrid files")
        
        # Verify model
        if os.path.exists(output_model):
            size_mb = os.path.getsize(output_model) / (1024*1024)
            print(f"✓ Acoustic model: {size_mb:.2f} MB")
        
        # Show sample TextGrid
        if textgrids:
            sample = textgrids[0]
            print(f"\nSample TextGrid ({sample}):")
            print("-"*60)
            with open(f"{mfa_output}/{sample}", 'r', encoding='utf-8') as f:
                content = f.read()
                print(content[:1200] if len(content) > 1200 else content)
                if len(content) > 1200:
                    print("...(truncated)")
        
        print("\n" + "="*60)
        print("✓ READY FOR NEXT STEP: FEATURE EXTRACTION")
        print("="*60)
        print(f"\nCompleted at: {subprocess.check_output(['date']).decode().strip()}")
        
    else:
        print(f"\n✗ MFA failed with return code {process.returncode}")
        print("\nCheck the error above. Common issues:")
        print("- Audio file format problems")
        print("- Text encoding issues")
        print("- Out of memory")
        print("- Missing conda environment (run: conda activate mfa)")
        
except KeyboardInterrupt:
    print("\n\n⚠️ Training interrupted!")
    process.terminate()
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()