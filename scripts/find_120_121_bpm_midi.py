import os
from pathlib import Path
import pretty_midi
import shutil

# Timothy Hyde 2025
# This script was used to scan through all my MIDI files in my dataset to find MIDI files which a tempo of 120 or 121. 
# These tempos are the default for MIDI files, so if any files had this tempo, it means they are PERFORMANCE BASED 
# MIDI files, not NOTATION BASED. I need notation based files only, since they adhere to timing and key signatures,
# whereas performance based MIDI files have notes not on the grid, with no time signature information for the model to learn from.


MIDI_INPUT_DIR = Path(r"C:\Users\school\Desktop\kunstderfuge_midi_sorted")
MIDI_OUTPUT_DIR = Path(r"C:\Users\school\Desktop\midi_120bpm_only2")    
MIDI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



for midi_file in MIDI_INPUT_DIR.glob("*.mid"):
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_file))
        tempos = midi.get_tempo_changes()[1]  

        # 120 or 121
        if all(int(round(t)) in {120, 121} for t in tempos):
            dest_path = MIDI_OUTPUT_DIR / midi_file.name
            shutil.move(str(midi_file), str(dest_path))
            print(f"Moved: {midi_file.name} ({set(map(int, map(round, tempos)))})")
        else:
            print(f"Skipped (tempos: {set(map(int, map(round, tempos)))}) — {midi_file.name}")

    except Exception as e:
        print(f" Error reading {midi_file.name}: {e}")

print("Complete")
