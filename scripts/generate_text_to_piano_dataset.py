from pathlib import Path
import json
from miditok import REMI
from miditoolkit import MidiFile
from tqdm import tqdm 
from pathlib import Path

# Timothy Hyde 2025
# This file was used for creating my custom dataset. It takes the MIDI information from the MAESTRO dataset json file,
# and then adds a space for my custom text descriptions.  

# Initializing
midi_dir = Path("/Users/timhyde/Desktop/MUSICGEN/midi")  
midi_files = list(midi_dir.glob("*.midi"))
tokenizer = REMI()

# Loading the maestro MIDI files
with open("/Users/timhyde/Desktop/MUSICGEN/maestro-v3.0.0.json", "r") as f:
    maestro_data = json.load(f)

# Creating list of dictionaries using the metadata
metadata_lookup = {}
for idx in maestro_data["midi_filename"]:
    full_path = maestro_data["midi_filename"][idx]
    stem = Path(full_path).stem
    metadata_lookup[stem] = {
        "composer": maestro_data["canonical_composer"].get(idx, "Unknown"),
        "title": maestro_data["canonical_title"].get(idx, "Untitled")
    }

# Generating a placeholder caption for each MIDI file. Displays the MIDI file name along with 
# the composer name and the piece name. Need to hand label each caption.
# Trying to figure out a way to scrape the internet for captions
def generate_caption(midi_path):
    midi_stem = midi_path.stem
    metadata = metadata_lookup.get(midi_stem)

    if metadata:
        composer = metadata["composer"]
        title = metadata["title"]
        return f"{midi_stem} INSERT_DESCRIPTION'{title}' by {composer}"
    else:
        return f"{midi_stem} INSERT_DESCRIPTION"

entries = []

for i, midi_path in enumerate(tqdm(midi_files, desc="Tokenizing MIDI files")):
    try:
        midi = MidiFile(midi_path)
        token_seq = tokenizer(midi)[0]
        tokens = token_seq.ids
        caption = generate_caption(midi_path)
        entries.append({
            "caption": caption,
            "midi_tokens": tokens
        })
    except Exception as e:
        print(f"Error with {midi_path.name}: {e}")

# changing MIDI Tokens into strings for easier storage
for entry in entries:
    entry["midi_tokens"] = "[" + ", ".join(map(str, entry["midi_tokens"])) + "]"

# Outputting
output_path = Path("/Users/timhyde/Desktop/MUSICGEN") / "text_to_piano_dataset_compact.json"
with open(output_path, "w") as f:
    json.dump(entries, f, indent=2, ensure_ascii=False)

print(f"Saved at location: {output_path}")
