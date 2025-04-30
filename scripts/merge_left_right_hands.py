import os
from pathlib import Path
from miditoolkit import MidiFile, Instrument

# Timothy Hyde 2025
# Some MIDI piano files seperate the two hands into two MIDI tracks, 
# and some even seperate the specific "voices", which could be 3 or 4 tracks.
# This script combines them into one single instrument track.

input_folder = Path(r"C:\Users\school\Desktop\DATA")
output_folder = input_folder / "merged"
output_folder.mkdir(exist_ok=True)


def merge_hands(midi_path, output_path, program=0):
    try:
        midi = MidiFile(midi_path)
    except Exception as e:
        print(f"Failed to load {midi_path.name}: {e}")
        return

    merged_notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            merged_notes.extend(inst.notes)

    if not merged_notes:
        print(f"No notes to merge in: {midi_path.name}")
        return

    merged_instrument = Instrument(program=program, is_drum=False)
    merged_instrument.notes = sorted(merged_notes, key=lambda n: n.start)
    midi.instruments = [merged_instrument]

    midi.dump(output_path)
    print(f"Merged MIDI saved to: {output_path}")




for midi_file in input_folder.glob("*.mid"):
    output_path = output_folder / midi_file.name
    merge_hands(midi_file, output_path)
