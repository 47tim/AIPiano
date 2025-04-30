import os
import re
from pathlib import Path


# Timothy Hyde 2025
# This script was used to renumber all the MIDI files in my dataset


MIDI_DIR = Path(r"C:\Users\school\Desktop\DATA")


def get_base_name(filename):
    return "_".join(filename.name.split("_")[1:])



def rename_all_midis_sequentially():
    midi_files = sorted(MIDI_DIR.glob("*.mid"))

    print(f"🎼 Found {len(midi_files)} MIDI files to rename.")

    for i, file in enumerate(midi_files, start=1):
        base_name = get_base_name(file)
        new_name = f"{i:04}_{base_name}"
        new_path = file.with_name(new_name)

        if file.name == new_path.name:
            print(f"Already correct: {file.name}")
            continue

        print(f"Renaming {file.name} → {new_name}")
        file.rename(new_path)

    print("Complete")

if __name__ == "__main__":
    rename_all_midis_sequentially()
