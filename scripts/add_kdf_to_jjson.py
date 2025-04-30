import json
from pathlib import Path

# Timothy Hyde 2025
# This script was used to add my new MIDI files from KDF to my existing JSON

json_path = Path(r"C:\Users\school\Desktop\Music Gen Master\MUSICGEN\final_new_final_text_to_piano_dataset_captions.json")
midi_folder = Path(r"C:\Users\school\Desktop\kdf_merged")
output_json_path = Path(r"C:\Users\school\Desktop\Music Gen Master\MUSICGEN\all_captions_datset.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)


midi_files = list(midi_folder.glob("*.mid"))

for midi_file in sorted(midi_files):
    filename = midi_file.stem  
    stripped_name = filename.split("_", 1)[1] if "_" in filename else filename
    new_entry = {"caption": stripped_name.replace("_", " ")} 
    data.append(new_entry)



with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Appended {len(midi_files)} entries and saved to: {output_json_path}")
