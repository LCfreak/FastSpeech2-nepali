import re
import glob
import os

# CHANGE THIS to your transcript folder
TRANSCRIPT_DIR = "Users/ashutoshbhattarai/Downloads/FastSpeech2/text_folder"

# Output dictionary filename
OUTPUT_DICT = "Users/ashutoshbhattarai/Downloads/FastSpeech2/nepali_mfa_dict.txt"

# Pattern for Devanagari words
devanagari = r"[\u0900-\u097F]+"

all_words = set()

# Get all .txt files from folder
txt_files = glob.glob(os.path.join(TRANSCRIPT_DIR, "*.txt"))

print(f"Found {len(txt_files)} transcript files.")

for file_path in txt_files:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        words = re.findall(devanagari, text)
        all_words.update(words)

def split_graphemes(word):
    return " ".join(list(word))

with open(OUTPUT_DICT, "w", encoding="utf-8") as out:
    out.write("!SIL SIL\n")
    out.write("<unk> SPN\n")
    for word in sorted(all_words):
        out.write(f"{word}\t{split_graphemes(word)}\n")

print("Dictionary saved to:", OUTPUT_DICT)
print("Total unique words:", len(all_words))
