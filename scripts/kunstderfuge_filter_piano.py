import os
from pathlib import Path
import pretty_midi
import shutil

# Timothy Hyde 2025
# This script was used for scanning through my KDF MIDI files, and filtering out all files which
# don't contain Piano only instruments.

MIDI_INPUT_DIR = Path(r"C:\Users\school\Desktop\kdf_unsorted_2")
MIDI_OUTPUT_DIR = Path(r"C:\Users\school\Desktop\kdf_sorted_2")
MIDI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_piano(instr: pretty_midi.Instrument):
    return not instr.is_drum and 0 <= instr.program <= 7



for midi_file in MIDI_INPUT_DIR.glob("*.mid"):
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_file))
        instruments = midi.instruments

        if len(instruments) == 1 and is_piano(instruments[0]):
            dest_path = MIDI_OUTPUT_DIR / midi_file.name
            shutil.copy(str(midi_file), str(dest_path))
            print(f"Copied solo piano: {midi_file.name}")

        elif len(instruments) == 2 and all(is_piano(i) for i in instruments):
            dest_path = MIDI_OUTPUT_DIR / midi_file.name
            shutil.copy(str(midi_file), str(dest_path))
            print(f"Copied piano duet: {midi_file.name}")

        else:
            print(f"Skipped: {midi_file.name}")

    except Exception as e:
        print(f"Error reading {midi_file.name}: {e}")

print("Complete")
