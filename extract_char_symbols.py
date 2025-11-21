import os
import re

base_path = "/Users/ashutoshbhattarai/Downloads/FastSpeech2"
mfa_output = os.path.join(base_path, "MFA_filelist")

print("="*60)
print("EXTRACTING CHARACTER-LEVEL SYMBOLS FROM TEXTGRIDS")
print("="*60)

# Get all TextGrid files
textgrid_files = [f for f in os.listdir(mfa_output) if f.endswith('.TextGrid')]
print(f"\nFound {len(textgrid_files)} TextGrid files")

# Extract all unique characters from phones tier
all_chars = set()
unk_count = 0
total_intervals = 0

print("\nProcessing TextGrids...")
for tg_file in textgrid_files[:min(100, len(textgrid_files))]:  # Sample first 100
    tg_path = os.path.join(mfa_output, tg_file)
    
    with open(tg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find phones tier section
    if 'name = "phones"' in content:
        # Extract text values from phones tier
        texts = re.findall(r'text = "(.*?)"', content)
        
        for text in texts:
            total_intervals += 1
            if text and text.strip():
                if text == '<unk>':
                    unk_count += 1
                elif text not in ['', 'SIL', 'SPN', 'spn']:
                    all_chars.add(text)

print(f"\n✓ Processed {min(100, len(textgrid_files))} TextGrids")
print(f"  Total intervals: {total_intervals}")
print(f"  Unique characters: {len(all_chars)}")
print(f"  <unk> tokens: {unk_count}")

# Show sample characters
print(f"\nSample characters (first 50):")
sample_chars = sorted(list(all_chars))[:50]
for i, char in enumerate(sample_chars):
    print(f"  {char}", end="  ")
    if (i + 1) % 10 == 0:
        print()

print("\n\nAll unique characters:")
print(sorted(list(all_chars)))

# Check if these are characters or words
avg_len = sum(len(c) for c in all_chars) / len(all_chars) if all_chars else 0
print(f"\nAverage character length: {avg_len:.2f}")

if avg_len > 1.5:
    print("⚠️ WARNING: These look like WORDS, not CHARACTERS!")
    print("Your MFA alignment might be using word-level instead of character-level")
elif avg_len <= 1.5:
    print("✓ GOOD: These are individual characters (graphemes)")

# Create symbols module
print("\n" + "="*60)
print("CREATING CHARACTER-LEVEL SYMBOLS MODULE")
print("="*60)

nepali_symbols_content = f'''# Nepali Character-Level Symbols for FastSpeech2
# Extracted from MFA TextGrid phones tier

# Special symbols
_pad = '_'
_eos = '~'
_unk = '?'

# Nepali characters (graphemes)
nepali_chars = {sorted(list(all_chars))}

# Create valid symbols list
valid_symbols = [_pad, _eos, _unk] + nepali_chars

# Create symbol to ID mapping
_symbol_to_id = {{s: i for i, s in enumerate(valid_symbols)}}
_id_to_symbol = {{i: s for i, s in enumerate(valid_symbols)}}

def text_to_sequence(text):
    """Convert text to sequence of symbol IDs"""
    sequence = []
    for char in text:
        if char in _symbol_to_id:
            sequence.append(_symbol_to_id[char])
        else:
            sequence.append(_symbol_to_id[_unk])  # Unknown character
    sequence.append(_symbol_to_id[_eos])  # Add EOS
    return sequence

def sequence_to_text(sequence):
    """Convert sequence of symbol IDs back to text"""
    result = []
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            if s != _eos and s != _pad:
                result.append(s)
    return ''.join(result)

print(f"Loaded {{len(valid_symbols)}} symbols ({{len(nepali_chars)}} Nepali chars + 3 special)")
'''

# Save files
text_dir = os.path.join(base_path, "text")
os.makedirs(text_dir, exist_ok=True)

symbols_path = os.path.join(text_dir, "nepali_symbols.py")
with open(symbols_path, 'w', encoding='utf-8') as f:
    f.write(nepali_symbols_content)

print(f"✓ Created: {symbols_path}")

# Update __init__.py
init_path = os.path.join(text_dir, "__init__.py")
init_content = '''from text.nepali_symbols import valid_symbols, text_to_sequence, sequence_to_text

__all__ = ['valid_symbols', 'text_to_sequence', 'sequence_to_text']
'''

with open(init_path, 'w', encoding='utf-8') as f:
    f.write(init_content)

print(f"✓ Updated: {init_path}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total symbols: {len(all_chars) + 3}")
print(f"Character-level: {avg_len <= 1.5}")

if avg_len > 1.5:
    print("\n⚠️ ACTION NEEDED:")
    print("Your MFA used word-level alignment, not character-level!")
    print("You may need to re-run MFA with proper character-level setup")
else:
    print("\n✓ Ready for preprocessing!")