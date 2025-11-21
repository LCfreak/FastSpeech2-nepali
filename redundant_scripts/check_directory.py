import os

# Update this path
base_path = "/Users/ashutoshbhattarai/Downloads/FastSpeech2"  # CHANGE THIS

print("="*60)
print("VERIFYING DIRECTORY STRUCTURE")
print("="*60)

required_dirs = {
    'mfa_input': os.path.join(base_path, 'mfa_input'),
    'MFA_filelist': os.path.join(base_path, 'MFA_filelist'),
    'wavs': os.path.join(base_path, 'wavs'),
    'data': os.path.join(base_path, 'data', 'SLR43', 'data'),
}

for name, path in required_dirs.items():
    exists = os.path.exists(path)
    print(f"{'✓' if exists else '✗'} {name}: {path}")
    if exists:
        files = os.listdir(path)
        if name == 'MFA_filelist':
            textgrids = [f for f in files if f.endswith('.TextGrid')]
            print(f"    TextGrids: {len(textgrids)}")
        elif name == 'wavs':
            wavs = [f for f in files if f.endswith('.wav')]
            print(f"    WAV files: {len(wavs)}")

# Check if preprocess.py exists
scripts = ['preprocess.py', 'train.py', 'hparams.py']
print("\n" + "="*60)
print("CHECKING REPO FILES")
print("="*60)

for script in scripts:
    path = os.path.join(base_path, script)
    exists = os.path.exists(path)
    print(f"{'✓' if exists else '✗'} {script}")

# Check metadata
metadata_path = os.path.join(base_path, 'data', 'SLR43', 'data', 'metadata.csv')
if os.path.exists(metadata_path):
    print(f"\n✓ metadata.csv found")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"    Total entries: {len(lines)}")
else:
    print(f"\n✗ metadata.csv not found at {metadata_path}")