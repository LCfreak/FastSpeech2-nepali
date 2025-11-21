import os
import re

base_path = "/Users/ashutoshbhattarai/Downloads/FastSpeech2"
mfa_output = os.path.join(base_path, "MFA_filelist")

print("="*60)
print("EXTRACTING ONLY FROM PHONES TIER")
print("="*60)

# Get all TextGrid files
textgrid_files = [f for f in os.listdir(mfa_output) if f.endswith('.TextGrid')]
print(f"Found {len(textgrid_files)} TextGrid files")

# Extract ONLY from phones tier
all_phones = set()
unk_count = 0

for tg_file in textgrid_files:
    tg_path = os.path.join(mfa_output, tg_file)
    
    with open(tg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by tiers
    sections = content.split('item [')
    
    # Find the phones tier specifically
    for section in sections:
        if 'name = "phones"' in section:
            # Extract text values ONLY from this section
            texts = re.findall(r'text = "(.*?)"', section)
            
            for text in texts:
                if text and text.strip():
                    if text == '<unk>':
                        unk_count += 1
                    elif text not in ['', 'SIL', 'SPN', 'spn']:
                        all_phones.add(text)
            break  # Stop after finding phones tier

print(f"\n✓ Extracted characters from phones tier only")
print(f"  Unique characters: {len(all_phones)}")
print(f"  <unk> tokens: {unk_count}")

# Check average length
avg_len = sum(len(p) for p in all_phones) / len(all_phones) if all_phones else 0
print(f"  Average character length: {avg_len:.2f}")

if avg_len <= 1.5:
    print("\n✅ CORRECT: Character-level symbols!")
else:
    print("\n⚠️ Still seeing multi-character symbols")

# Show all characters
print(f"\nAll unique characters ({len(all_phones)}):")
sorted_phones = sorted(list(all_phones))
print(sorted_phones)

# Create symbols module
print("\n" + "="*60)
print("CREATING CHARACTER SYMBOLS MODULE")
print("="*60)

nepali_symbols_content = f'''# Nepali Character-Level Symbols
# Extracted from MFA TextGrid phones tier only

# Special symbols
_pad = '_'
_eos = '~'
_unk = '?'

# Nepali characters (from phones tier)
nepali_chars = {sorted_phones}

# Create valid symbols list
valid_symbols = [_pad, _eos, _unk] + nepali_chars

# Symbol to ID mapping
_symbol_to_id = {{s: i for i, s in enumerate(valid_symbols)}}
_id_to_symbol = {{i: s for i, s in enumerate(valid_symbols)}}

def text_to_sequence(text):
    """Convert text to sequence of symbol IDs"""
    sequence = []
    for char in text:
        if char in _symbol_to_id:
            sequence.append(_symbol_to_id[char])
        else:
            sequence.append(_symbol_to_id[_unk])
    sequence.append(_symbol_to_id[_eos])
    return sequence

def sequence_to_text(sequence):
    """Convert sequence back to text"""
    result = []
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            if s not in [_eos, _pad]:
                result.append(s)
    return ''.join(result)

print(f"Loaded {{len(valid_symbols)}} symbols")
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
print(f"Total symbols: {len(all_phones) + 3}")
print(f"Character-level: {avg_len <= 1.5}")

if avg_len <= 1.5:
    print("\n✅ Ready for preprocessing!")
else:
    print("\n⚠️ Need to check TextGrid format")