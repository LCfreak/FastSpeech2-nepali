import os
from pathlib import Path

# Update these paths to match your local PC structure
base_path = '/Users/ashutoshbhattarai/Downloads/FastSpeech2'  # CHANGE THIS
mfa_output = os.path.join(base_path, 'MFA_filelist')
mfa_input = os.path.join(base_path, 'mfa_input')

print("="*60)
print("DIAGNOSING MFA ALIGNMENT RESULTS")
print("="*60)

# Count files
lab_files = [f.replace('.lab', '') for f in os.listdir(mfa_input) if f.endswith('.lab')]
textgrid_files = [f.replace('.TextGrid', '') for f in os.listdir(mfa_output) if f.endswith('.TextGrid')]

print(f"\nInput .lab files: {len(lab_files)}")
print(f"Output TextGrids: {len(textgrid_files)}")
print(f"Success rate: {len(textgrid_files)/len(lab_files)*100:.1f}%")
print(f"Missing: {len(lab_files) - len(textgrid_files)}")

# Find which files are missing
missing = set(lab_files) - set(textgrid_files)

if len(missing) > 0:
    print(f"\nMissing files: {len(missing)}")
    print("\nFirst 20 missing files:")
    for i, f in enumerate(list(missing)[:20]):
        print(f"  {f}")
    
    # Save missing files list
    missing_file = os.path.join(base_path, 'missing_files.txt')
    with open(missing_file, 'w', encoding='utf-8') as f:
        for filename in sorted(missing):
            f.write(f"{filename}\n")
    print(f"\n✓ Saved all missing files to: {missing_file}")

# Check for <unk> tokens in TextGrids
print("\n" + "="*60)
print("CHECKING FOR <unk> TOKENS")
print("="*60)

unk_count = 0
total_unk_instances = 0
files_with_unk = []

sample_size = min(50, len(textgrid_files))  # Check first 50
sample_textgrids = [f for f in os.listdir(mfa_output) if f.endswith('.TextGrid')][:sample_size]

for tg in sample_textgrids:
    with open(os.path.join(mfa_output, tg), 'r', encoding='utf-8') as f:
        content = f.read()
        unk_in_file = content.count('<unk>')
        if unk_in_file > 0:
            unk_count += 1
            total_unk_instances += unk_in_file
            files_with_unk.append((tg, unk_in_file))

print(f"\nSampled {sample_size} TextGrids:")
print(f"  Files with <unk>: {unk_count}/{sample_size} ({unk_count/sample_size*100:.1f}%)")
print(f"  Total <unk> instances: {total_unk_instances}")

if files_with_unk:
    print(f"\nFiles with most <unk> tokens:")
    for tg, count in sorted(files_with_unk, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {tg}: {count} <unk> tokens")

# Check a sample TextGrid in detail
if len(textgrid_files) > 0:
    print("\n" + "="*60)
    print("SAMPLE TEXTGRID ANALYSIS")
    print("="*60)
    
    sample = sample_textgrids[0]
    print(f"\nAnalyzing: {sample}")
    
    with open(os.path.join(mfa_output, sample), 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Extract phones tier
    if 'name = "phones"' in content:
        print("✓ Phones tier found")
        # Count actual characters vs <unk>
        import re
        phone_texts = re.findall(r'text = "(.*?)"', content)
        actual_chars = [t for t in phone_texts if t and t != '' and t != '<unk>']
        unk_tokens = [t for t in phone_texts if t == '<unk>']
        
        print(f"  Total intervals: {len(phone_texts)}")
        print(f"  Actual characters: {len(actual_chars)}")
        print(f"  <unk> tokens: {len(unk_tokens)}")
        
        if len(actual_chars) > 0:
            print(f"\n  Sample characters found: {actual_chars[:20]}")

# Final assessment
print("\n" + "="*60)
print("ASSESSMENT")
print("="*60)

success_rate = len(textgrid_files)/len(lab_files)*100
unk_rate = unk_count/sample_size*100 if sample_size > 0 else 0

if success_rate < 50:
    print("❌ CRITICAL: Less than 50% files aligned successfully")
    print("   Action: Need to fix dictionary and re-run MFA")
elif success_rate < 80:
    print("⚠️  WARNING: Less than 80% files aligned")
    print("   Action: Recommended to fix dictionary and re-run")
elif unk_rate > 50:
    print("⚠️  WARNING: More than 50% of TextGrids contain <unk>")
    print("   Action: Dictionary is incomplete, should fix")
else:
    print("✅ ACCEPTABLE: Can proceed with feature extraction")
    print(f"   You have {len(textgrid_files)} usable samples")

print("\n" + "="*60)