#!/usr/bin/env python3
"""
Script to inspect TextGrid structure and character-level alignments
"""
import sys
import os

def read_textgrid(filepath):
    """Read and display TextGrid structure"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"\n{'='*60}")
    print(f"TextGrid File: {os.path.basename(filepath)}")
    print(f"{'='*60}\n")
    
    # Find basic info
    for i, line in enumerate(lines[:10]):
        if 'xmax' in line:
            duration = line.split('=')[1].strip()
            print(f"Duration: {duration} seconds")
        if 'size' in line and 'tiers' not in line:
            num_tiers = line.split('=')[1].strip()
            print(f"Number of tiers: {num_tiers}")
    
    print("\n" + "-"*60)
    
    # Find tier information
    tier_count = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if 'item [' in line and ']:' in line:
            tier_count += 1
            print(f"\n{'='*60}")
            print(f"TIER {tier_count}")
            print(f"{'='*60}")
            
            # Get tier details
            for j in range(i, min(i+20, len(lines))):
                if 'name =' in lines[j]:
                    tier_name = lines[j].split('=')[1].strip().strip('"')
                    print(f"Tier Name: {tier_name}")
                if 'intervals: size =' in lines[j]:
                    num_intervals = lines[j].split('=')[1].strip()
                    print(f"Number of intervals: {num_intervals}")
                    break
            
            # Show first 5 intervals
            print(f"\nFirst 5 intervals:")
            print("-"*60)
            interval_count = 0
            k = i
            while k < len(lines) and interval_count < 5:
                if 'intervals [' in lines[k] and ']:' in lines[k]:
                    interval_count += 1
                    xmin = xmax = text = ""
                    for m in range(k, min(k+10, len(lines))):
                        if 'xmin' in lines[m] and '=' in lines[m]:
                            xmin = lines[m].split('=')[1].strip()
                        if 'xmax' in lines[m] and '=' in lines[m]:
                            xmax = lines[m].split('=')[1].strip()
                        if 'text' in lines[m] and '=' in lines[m]:
                            text = lines[m].split('=')[1].strip().strip('"')
                            break
                    
                    duration_ms = (float(xmax) - float(xmin)) * 1000
                    print(f"  [{interval_count}] {xmin:>6} -> {xmax:>6} ({duration_ms:>6.1f}ms) : '{text}'")
                k += 1
        
        i += 1
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Total tiers found: {tier_count}")
    
    # Check for character tier
    has_char_tier = False
    for line in lines:
        if 'name =' in line:
            tier_name = line.split('=')[1].strip().strip('"').lower()
            if tier_name in ['phones', 'characters', 'chars', 'graphemes']:
                has_char_tier = True
                print(f"✓ Character-level tier found: '{tier_name}'")
    
    if not has_char_tier:
        print("⚠ WARNING: No character-level tier found!")
        print("  Expected tier names: 'phones', 'characters', 'chars', or 'graphemes'")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_textgrid.py <textgrid_file>")
        print("\nExample:")
        print("  python inspect_textgrid.py MFA_filelist/nep_0258_0119737288.TextGrid")
        sys.exit(1)
    
    textgrid_path = sys.argv[1]
    
    if not os.path.exists(textgrid_path):
        print(f"Error: File not found: {textgrid_path}")
        sys.exit(1)
    
    read_textgrid(textgrid_path)