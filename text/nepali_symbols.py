# Nepali Phoneme-Level Symbols for FastSpeech2

# Special symbols
_pad = '_'
_eos = '~'
_unk = '?'

# Individual phonemes extracted from MFA dictionary
phonemes = ['ँ', 'ं', 'ः', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', '्', 'ॠ', '८']

# Create valid symbols list
valid_symbols = [_pad, _eos, _unk] + phonemes

# Create symbol to ID mapping
_symbol_to_id = {s: i for i, s in enumerate(valid_symbols)}
_id_to_symbol = {i: s for i, s in enumerate(valid_symbols)}

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

print(f"Loaded {len(valid_symbols)} symbols ({len(phonemes)} unique phonemes + 3 special tokens)")
