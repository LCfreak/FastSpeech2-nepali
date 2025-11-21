# import os
# import glob
# import tqdm
# import torch
# import argparse
# import numpy as np
# import hparams as hp
# from stft import TacotronSTFT
# from utils.utils import read_wav_np
# from audio_processing import pitch
# from text import phonemes_to_sequence

# def main(args):
#     stft = TacotronSTFT(filter_length=hp.n_fft,
#                         hop_length=hp.hop_length,
#                         win_length=hp.win_length,
#                         n_mel_channels=hp.n_mels,
#                         sampling_rate=hp.sampling_rate,
#                         mel_fmin=hp.fmin,
#                         mel_fmax=hp.fmax)
#     # wav_file loacation 
#     wav_files = glob.glob(os.path.join(args.wav_root_path, '**', '*.wav'), recursive=True)
    
#     #Define all the paths correesponding to the feature
#     text_path = os.path.join(hp.data_path, 'text')
#     mel_path = os.path.join(hp.data_path, 'mels')
#     duration_path = os.path.join(hp.data_path, 'alignment')
#     energy_path = os.path.join(hp.data_path, 'energy')
#     pitch_path = os.path.join(hp.data_path, 'pitch')
#     symbol_path = os.path.join(hp.data_path, 'symbol')
    
#     # create directory if doesnt exist
#     os.makedirs(text_path,exist_ok = True)
#     os.makedirs(duration_path, exist_ok = True)
#     os.makedirs(mel_path, exist_ok=True)
#     os.makedirs(energy_path, exist_ok=True)
#     os.makedirs(pitch_path, exist_ok=True)
#     os.makedirs(symbol_path, exist_ok=True)
    
#     for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to mel, energy, and pitch'):
#         sr, wav = read_wav_np(wavpath)
#         p = pitch(wav)  # [T, ] T = Number of frames
#         wav = torch.from_numpy(wav).unsqueeze(0)      
#         mel, mag = stft.mel_spectrogram(wav) # mel [1, 80, T]  mag [1, num_mag, T]
#         mel = mel.squeeze(0) # [num_mel, T]
#         mag = mag.squeeze(0) # [num_mag, T]
#         e = torch.norm(mag, dim=0) # [T, ]
#         p = p[:mel.shape[1]]
#         p = np.array(p, dtype='float32')
#         id = os.path.basename(wavpath).split(".")[0]
        
#         # save the features
#         np.save('{}/{}.npy'.format(mel_path,id), mel.numpy(), allow_pickle=False)
#         np.save('{}/{}.npy'.format(energy_path, id), e.numpy(), allow_pickle=False)
#         np.save('{}/{}.npy'.format(pitch_path, id), p , allow_pickle=False)
        
        
    
#     with open(hp.filelist_alignment_dir + "alignment.txt", encoding='utf-8') as f:      #add all 13100 examples to filelist.txt 
#         for lines in f:
#             content = lines.split('|')
#             id = content[4].split()[0].split('.')[0]
#             if os.path.exists(os.path.join(args.wav_root_path, id + '.wav')):
#                 text = content[0]
#                 duration = content[2]
#                 duration = duration.split()
#                 dur = np.array(duration, dtype = 'float32')         #done
#                 phoneme = content[3]
#                 symbol_sequence = phonemes_to_sequence(phoneme)      
            
#                 np.save('{}/{}.npy'.format(text_path, id), (text, phoneme), allow_pickle=False) #what is the input text or phonemen???
#                 np.save('{}/{}.npy'.format(duration_path, id), dur, allow_pickle=False)
#                 np.save('{}/{}.npy'.format(symbol_path, id), symbol_sequence, allow_pickle=False)
            


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', '--wav_root_path', type=str, required=True,
#                         help="root directory of wav files")
#     args = parser.parse_args()

#     main(args)





"""
Nepali FastSpeech2 Preprocessing Script
Extracts durations, pitch, energy from MFA TextGrids and audio files
"""

import os
import json
import argparse
import numpy as np
import pyworld as pw
import librosa
import textgrid
from tqdm import tqdm
from scipy.interpolate import interp1d
import soundfile as sf
from pathlib import Path


class NepaliPreprocessor:
    def __init__(self, 
                 textgrid_dir="MFA_filelist",
                 wav_dir="wavs",
                 text_dir="text",
                 output_dir="preprocessed_data/nepali",
                 sample_rate=22050,
                 hop_length=256,
                 n_fft=1024,
                 win_length=1024,
                 n_mels=80,
                 fmin=0,
                 fmax=8000):
        
        self.textgrid_dir = textgrid_dir
        self.wav_dir = wav_dir
        self.text_dir = text_dir
        self.output_dir = output_dir
        
        # Audio parameters
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        
        # Create output directories
        self.create_output_dirs()
        
    def create_output_dirs(self):
        """Create necessary output directories"""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/duration",
            f"{self.output_dir}/pitch",
            f"{self.output_dir}/energy",
            f"{self.output_dir}/mel",
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def extract_duration_from_textgrid(self, textgrid_path):
        """
        Extract phoneme durations from TextGrid file
        Returns: phonemes list and durations array
        """
        try:
            tg = textgrid.TextGrid.fromFile(textgrid_path)
            
            # Find the phones tier (usually named "phones")
            phones_tier = None
            for tier in tg.tiers:
                if tier.name == "phones":
                    phones_tier = tier
                    break
            
            if phones_tier is None:
                raise ValueError(f"No 'phones' tier found in {textgrid_path}")
            
            phonemes = []
            durations = []
            
            for interval in phones_tier:
                phone = interval.mark.strip()
                if phone and phone != "":  # Skip empty intervals
                    duration = interval.maxTime - interval.minTime
                    # Convert duration from seconds to frames
                    duration_frames = int(np.round(duration * self.sample_rate / self.hop_length))
                    
                    phonemes.append(phone)
                    durations.append(duration_frames)
            
            return phonemes, np.array(durations, dtype=np.int32)
        
        except Exception as e:
            print(f"Error processing {textgrid_path}: {str(e)}")
            return None, None
    
    def extract_mel_spectrogram(self, wav_path):
        """Extract mel-spectrogram from audio"""
        try:
            # Load audio
            wav, sr = sf.read(wav_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
            
            # Compute mel-spectrogram
            mel = librosa.feature.melspectrogram(
                y=wav,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax
            )
            
            # Convert to log scale
            mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
            
            return mel.T  # Shape: (T, n_mels)
        
        except Exception as e:
            print(f"Error extracting mel from {wav_path}: {str(e)}")
            return None
    
    def extract_pitch(self, wav_path):
        """
        Extract pitch (F0) using PyWorld
        """
        try:
            # Load audio
            wav, sr = sf.read(wav_path)
            wav = wav.astype(np.float64)
            
            # Resample if necessary
            if sr != self.sample_rate:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
            
            # Extract pitch using PyWorld
            f0, timeaxis = pw.dio(
                wav,
                self.sample_rate,
                frame_period=self.hop_length / self.sample_rate * 1000
            )
            f0 = pw.stonemask(wav, f0, timeaxis, self.sample_rate)
            
            return f0
        
        except Exception as e:
            print(f"Error extracting pitch from {wav_path}: {str(e)}")
            return None
    
    def extract_energy(self, mel):
        """
        Extract energy from mel-spectrogram
        Energy is computed as the L2-norm of mel frames
        """
        energy = np.linalg.norm(mel, axis=1)
        return energy
    
    def process_utterance(self, basename):
        """
        Process a single utterance
        """
        textgrid_path = os.path.join(self.textgrid_dir, f"{basename}.TextGrid")
        wav_path = os.path.join(self.wav_dir, f"{basename}.wav")
        
        # Check if files exist
        if not os.path.exists(textgrid_path):
            print(f"TextGrid not found: {textgrid_path}")
            return None
        if not os.path.exists(wav_path):
            print(f"WAV not found: {wav_path}")
            return None
        
        # Extract duration from TextGrid
        phonemes, durations = self.extract_duration_from_textgrid(textgrid_path)
        if phonemes is None or durations is None:
            return None
        
        # Extract mel-spectrogram
        mel = self.extract_mel_spectrogram(wav_path)
        if mel is None:
            return None
        
        # Extract pitch
        pitch = self.extract_pitch(wav_path)
        if pitch is None:
            return None
        
        # Extract energy from mel
        energy = self.extract_energy(mel)
        
        # Align lengths (mel, pitch, energy should have same length)
        min_len = min(len(mel), len(pitch), len(energy))
        mel = mel[:min_len]
        pitch = pitch[:min_len]
        energy = energy[:min_len]
        
        # Check if total duration matches mel length
        total_duration = sum(durations)
        if abs(total_duration - min_len) > 5:  # Allow small tolerance
            print(f"Warning: Duration mismatch for {basename}: {total_duration} vs {min_len}")
            # Adjust durations proportionally
            scale = min_len / total_duration
            durations = np.round(durations * scale).astype(np.int32)
            # Fix any rounding errors
            durations[-1] += min_len - sum(durations)
        
        # Save processed data
        np.save(f"{self.output_dir}/duration/{basename}.npy", durations)
        np.save(f"{self.output_dir}/pitch/{basename}.npy", pitch)
        np.save(f"{self.output_dir}/energy/{basename}.npy", energy)
        np.save(f"{self.output_dir}/mel/{basename}.npy", mel)
        
        return {
            "basename": basename,
            "phonemes": "|".join(phonemes),
            "n_frames": min_len,
            "duration_sum": int(sum(durations))
        }
    
    def process_all(self):
        """
        Process all utterances
        """
        # Get all TextGrid files
        textgrid_files = [f for f in os.listdir(self.textgrid_dir) if f.endswith(".TextGrid")]
        basenames = [f.replace(".TextGrid", "") for f in textgrid_files]
        
        print(f"Found {len(basenames)} files to process")
        
        metadata = []
        failed = []
        
        for basename in tqdm(basenames, desc="Processing utterances"):
            result = self.process_utterance(basename)
            if result is not None:
                metadata.append(result)
            else:
                failed.append(basename)
        
        print(f"\nSuccessfully processed: {len(metadata)}")
        print(f"Failed: {len(failed)}")
        
        if failed:
            with open(f"{self.output_dir}/failed_files.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(failed))
        
        # Save metadata
        with open(f"{self.output_dir}/metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return metadata
    
    def compute_statistics(self, metadata):
        """
        Compute statistics for pitch and energy normalization
        """
        print("\nComputing statistics...")
        
        all_pitch = []
        all_energy = []
        
        for item in tqdm(metadata, desc="Loading data"):
            basename = item["basename"]
            
            pitch = np.load(f"{self.output_dir}/pitch/{basename}.npy")
            energy = np.load(f"{self.output_dir}/energy/{basename}.npy")
            
            # Remove zeros from pitch (unvoiced frames)
            pitch_nonzero = pitch[pitch > 0]
            if len(pitch_nonzero) > 0:
                all_pitch.extend(pitch_nonzero)
            
            all_energy.extend(energy)
        
        all_pitch = np.array(all_pitch)
        all_energy = np.array(all_energy)
        
        stats = {
            "pitch": {
                "min": float(np.min(all_pitch)),
                "max": float(np.max(all_pitch)),
                "mean": float(np.mean(all_pitch)),
                "std": float(np.std(all_pitch))
            },
            "energy": {
                "min": float(np.min(all_energy)),
                "max": float(np.max(all_energy)),
                "mean": float(np.mean(all_energy)),
                "std": float(np.std(all_energy))
            }
        }
        
        # Save statistics
        with open(f"{self.output_dir}/stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print("\nStatistics:")
        print(f"Pitch  - Min: {stats['pitch']['min']:.2f}, Max: {stats['pitch']['max']:.2f}, "
              f"Mean: {stats['pitch']['mean']:.2f}, Std: {stats['pitch']['std']:.2f}")
        print(f"Energy - Min: {stats['energy']['min']:.2f}, Max: {stats['energy']['max']:.2f}, "
              f"Mean: {stats['energy']['mean']:.2f}, Std: {stats['energy']['std']:.2f}")
        
        return stats
    
    def create_filelists(self, metadata, val_ratio=0.05):
        """
        Create train/val filelists
        Format: basename|phonemes
        """
        print("\nCreating filelists...")
        
        np.random.seed(42)
        indices = np.arange(len(metadata))
        np.random.shuffle(indices)
        
        val_size = int(len(metadata) * val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        train_data = [metadata[i] for i in train_indices]
        val_data = [metadata[i] for i in val_indices]
        
        # Save train filelist
        with open(f"{self.output_dir}/train.txt", "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(f"{item['basename']}|{item['phonemes']}\n")
        
        # Save val filelist
        with open(f"{self.output_dir}/val.txt", "w", encoding="utf-8") as f:
            for item in val_data:
                f.write(f"{item['basename']}|{item['phonemes']}\n")
        
        print(f"Train samples: {len(train_data)}")
        print(f"Val samples: {len(val_data)}")
        
        return train_data, val_data


def main():
    parser = argparse.ArgumentParser(description="Preprocess Nepali dataset for FastSpeech2")
    parser.add_argument("--textgrid_dir", type=str, default="MFA_filelist",
                        help="Directory containing TextGrid files")
    parser.add_argument("--wav_dir", type=str, default="wavs",
                        help="Directory containing WAV files")
    parser.add_argument("--text_dir", type=str, default="text",
                        help="Directory containing text transcriptions")
    parser.add_argument("--output_dir", type=str, default="preprocessed_data/nepali",
                        help="Output directory for preprocessed data")
    parser.add_argument("--sample_rate", type=int, default=22050,
                        help="Audio sample rate")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Validation set ratio")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = NepaliPreprocessor(
        textgrid_dir=args.textgrid_dir,
        wav_dir=args.wav_dir,
        text_dir=args.text_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate
    )
    
    # Process all utterances
    metadata = preprocessor.process_all()
    
    if len(metadata) == 0:
        print("No data processed successfully. Exiting.")
        return
    
    # Compute statistics
    stats = preprocessor.compute_statistics(metadata)
    
    # Create filelists
    preprocessor.create_filelists(metadata, val_ratio=args.val_ratio)
    
    print("\nPreprocessing complete!")
    print(f"Preprocessed data saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Update your config files with the statistics from stats.json")
    print("2. Update phoneme symbols list")
    print("3. Start training with: python train.py")


if __name__ == "__main__":
    main()
