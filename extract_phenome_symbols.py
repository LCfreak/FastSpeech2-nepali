"""
Extract unique phoneme symbols from TextGrid files
Creates symbols.txt file for FastSpeech2
"""

import os
import textgrid
from collections import Counter
import argparse


def extract_phonemes_from_textgrid(textgrid_path):
    """Extract all phonemes from a single TextGrid file"""
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        
        # Find the phones tier
        phones_tier = None
        for tier in tg.tiers:
            if tier.name == "phones":
                phones_tier = tier
                break
        
        if phones_tier is None:
            return []
        
        phonemes = []
        for interval in phones_tier:
            phone = interval.mark.strip()
            if phone and phone != "":
                phonemes.append(phone)
        
        return phonemes
    
    except Exception as e:
        print(f"Error processing {textgrid_path}: {str(e)}")
        return []


def extract_all_phonemes(textgrid_dir):
    """Extract all unique phonemes from all TextGrid files"""
    all_phonemes = []
    phoneme_counts = Counter()
    
    textgrid_files = [f for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")]
    
    print(f"Processing {len(textgrid_files)} TextGrid files...")
    
    for tg_file in textgrid_files:
        tg_path = os.path.join(textgrid_dir, tg_file)
        phonemes = extract_phonemes_from_textgrid(tg_path)
        all_phonemes.extend(phonemes)
        phoneme_counts.update(phonemes)
    
    unique_phonemes = sorted(set(all_phonemes))
    
    return unique_phonemes, phoneme_counts


def create_symbols_file(phonemes, output_path="preprocessed_data/nepali/symbols.txt"):
    """
    Create symbols.txt file for FastSpeech2
    Format includes special tokens: _pad_, _eos_, _unk_ + phonemes
    """
    
    # Special tokens
    special_tokens = ["_pad_", "_eos_", "_unk_"]
    
    # Combine special tokens with phonemes
    all_symbols = special_tokens + phonemes
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for symbol in all_symbols:
            f.write(f"{symbol}\n")
    
    print(f"\nSymbols file created: {output_path}")
    print(f"Total symbols: {len(all_symbols)} (including {len(special_tokens)} special tokens)")
    
    return all_symbols


def print_phoneme_statistics(phonemes, phoneme_counts):
    """Print statistics about phonemes"""
    print(f"\nTotal unique phonemes: {len(phonemes)}")
    print(f"Total phoneme tokens: {sum(phoneme_counts.values())}")
    
    print("\nTop 20 most frequent phonemes:")
    for phone, count in phoneme_counts.most_common(20):
        print(f"  {phone}: {count}")
    
    print("\nAll unique phonemes:")
    print("  " + ", ".join(phonemes))


def main():
    parser = argparse.ArgumentParser(description="Extract phoneme symbols from TextGrid files")
    parser.add_argument("--textgrid_dir", type=str, default="MFA_filelist",
                        help="Directory containing TextGrid files")
    parser.add_argument("--output", type=str, default="preprocessed_data/nepali/symbols.txt",
                        help="Output path for symbols.txt")
    parser.add_argument("--stats", action="store_true",
                        help="Print detailed statistics")
    
    args = parser.parse_args()
    
    # Extract phonemes
    phonemes, phoneme_counts = extract_all_phonemes(args.textgrid_dir)
    
    if not phonemes:
        print("No phonemes found!")
        return
    
    # Print statistics
    if args.stats:
        print_phoneme_statistics(phonemes, phoneme_counts)
    else:
        print(f"Found {len(phonemes)} unique phonemes")
    
    # Create symbols file
    create_symbols_file(phonemes, args.output)
    
    # Also save as Python list for easy import
    py_output = args.output.replace(".txt", ".py")
    with open(py_output, "w", encoding="utf-8") as f:
        f.write("# Nepali phoneme symbols for FastSpeech2\n")
        f.write("# Auto-generated from TextGrid files\n\n")
        f.write("symbols = [\n")
        f.write("    '_pad_',\n")
        f.write("    '_eos_',\n")
        f.write("    '_unk_',\n")
        for phone in phonemes:
            f.write(f"    '{phone}',\n")
        f.write("]\n\n")
        f.write(f"# Total: {len(phonemes) + 3} symbols\n")
    
    print(f"Python symbols file created: {py_output}")


if __name__ == "__main__":
    main()