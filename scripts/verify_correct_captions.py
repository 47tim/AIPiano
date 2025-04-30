import json
import ast
from pathlib import Path
from miditok import REMI


# Timothy Hyde 2025
# This script was used to verify that my dataset and JSON order matched. 
# Checking to see if the captions match the file name and if the file matches the tokenized version in the JSON.


json_path = Path(r"C:\Users\school\Desktop\Music Gen Master\MUSICGEN\all_captions_cleaned.json")
midi_folder = Path(r"C:\Users\school\Desktop\Music Gen Master\MUSICGEN\DATA")
tokenizer_config_path = Path(r"C:\Users\school\Desktop\Music Gen Master\MUSICGEN\tokenizer_config\tokenizer.json")
tokenizer = REMI(params=tokenizer_config_path)

# === Load tokenizer from saved params ===
tokenizer = REMI(params=tokenizer_config_path)

# === Load MIDI files (sorted to match JSON order) ===
midi_files = sorted(midi_folder.glob("*.mid"))

# === Load JSON dataset ===
with open(json_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# === Sanity check ===
if len(midi_files) != len(dataset):
    print(f"⚠️ Mismatch: {len(midi_files)} MIDI files but {len(dataset)} JSON entries!")
    exit(1)

# === Verification loop ===
mismatches = []

for idx, (entry, midi_file) in enumerate(zip(dataset, midi_files)):
    tokens_stored = ast.literal_eval(entry["midi_tokens"]) if isinstance(entry["midi_tokens"], str) else entry["midi_tokens"]

    try:
        tokens_actual = tokenizer.encode(midi_file)[0].ids

        if tokens_stored == tokens_actual:
            print(f"✅ Match — {midi_file.name}")
            print(f"   Caption: {entry['caption']}")
        else:
            print(f"❌ Mismatch — {midi_file.name}")
            print(f"   Caption: {entry['caption']}")
            mismatches.append((midi_file.name, entry['caption'], tokens_stored[:10], tokens_actual[:10]))

    except Exception as e:
        print(f"⚠️ Error processing {midi_file.name}: {e}")
        print(f"   Caption: {entry['caption']}")
        mismatches.append((midi_file.name, entry['caption'], str(e)))

# === Summary ===
print("\n🏁 Verification complete.")
print(f"✅ Matches: {len(dataset) - len(mismatches)}")
print(f"❌ Mismatches: {len(mismatches)}")

if mismatches:
    print("\nTop mismatches:")
    for name, json_tokens, midi_tokens in mismatches[:5]:
        print(f"- {name}:")
        print(f"  → JSON: {json_tokens}")
        print(f"  → MIDI: {midi_tokens}")
