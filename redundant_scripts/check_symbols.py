import os

base_path = "/Users/ashutoshbhattarai/Downloads/FastSpeech2"
dict_path = os.path.join(base_path, "nepali_mfa_dict.txt")
text_symbols_path = os.path.join(base_path, "text", "__init__.py")

print("="*60)
print("EXTRACTING INDIVIDUAL PHONEMES")
print("="*60)

# Read dictionary and extract individual phonemes
phonemes = set()
with open(dict_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            word, phoneme_sequence = parts
            # Skip special tokens
            if word in ['!SIL', '<unk>']:
                continue
            # Split phoneme sequence and add each phoneme
            individual_phonemes = phoneme_sequence.split()
            phonemes.update(individual_phonemes)

print(f"\nTotal unique phonemes: {len(phonemes)}")
print(f"Sample phonemes: {sorted(list(phonemes))[:50]}")

# Create Nepali symbols file with phoneme-level symbols
nepali_symbols_content = f'''# Nepali Phoneme-Level Symbols for FastSpeech2

# Special symbols
_pad = '_'
_eos = '~'
_unk = '?'

# Individual phonemes extracted from MFA dictionary
phonemes = {sorted(list(phonemes))}

# Create valid symbols list
valid_symbols = [_pad, _eos, _unk] + phonemes

# Create symbol to ID mapping
_symbol_to_id = {{s: i for i, s in enumerate(valid_symbols)}}
_id_to_symbol = {{i: s for i, s in enumerate(valid_symbols)}}

def text_to_sequence(text):
    """Convert text string (space-separated phonemes) to sequence of IDs"""
    sequence = []
    phoneme_list = text.split()  # Split by space to get individual phonemes
    
    for phoneme in phoneme_list:
        if phoneme in _symbol_to_id:
            sequence.append(_symbol_to_id[phoneme])
        else:
            sequence.append(_symbol_to_id[_unk])  # Unknown phoneme
    
    sequence.append(_symbol_to_id[_eos])  # Add EOS
    return sequence

def sequence_to_text(sequence):
    """Convert sequence of IDs back to phoneme string"""
    result = []
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            if s != _eos and s != _pad:
                result.append(s)
    return ' '.join(result)  # Join with space for phoneme sequence

print(f"Loaded {{len(valid_symbols)}} symbols ({{len(phonemes)}} unique phonemes + 3 special tokens)")
'''

# Create text directory if it doesn't exist
symbols_output = os.path.join(base_path, "text", "nepali_symbols.py")
os.makedirs(os.path.dirname(symbols_output), exist_ok=True)

# Write the symbols file
with open(symbols_output, 'w', encoding='utf-8') as f:
    f.write(nepali_symbols_content)

print(f"\n✓ Created: {symbols_output}")
print(f"  Total symbols: {len(phonemes) + 3} (phonemes + special tokens)")

# Update text/__init__.py
init_content = '''from text.nepali_symbols import valid_symbols, text_to_sequence, sequence_to_text

__all__ = ['valid_symbols', 'text_to_sequence', 'sequence_to_text']
'''

with open(text_symbols_path, 'w', encoding='utf-8') as f:
    f.write(init_content)

print(f"✓ Updated: {text_symbols_path}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✓ Extracted {len(phonemes)} unique phonemes")
print(f"✓ Created phoneme-level symbol system")
print(f"✓ Each phoneme is now a separate symbol")
print("\nExample usage:")
print("  Input:  'अ ग ा ड ि'")
print("  Output: [symbol_ids for each phoneme]")