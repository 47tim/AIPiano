import json
from pathlib import Path

# Timothy Hyde 2025
# This file was used to merge the captions back in with the midi data, for the complete dataset.

with open("/Users/timhyde/Desktop/MUSICGEN/text_to_piano_dataset_compact.json", "r") as f:
    full_data = json.load(f)

with open("/Users/timhyde/Desktop/MUSICGEN/captions_cleaned.json", "r") as f:
    enriched_captions = json.load(f)

# Making sure the amount of captions matches the amount of MIDI sequences. Praying this works, really
# don't want to manually check for what's missing!
assert len(full_data) == len(enriched_captions), "Error: lengths do not match."

# Replacing captions with the new ones I created
for original, enriched in zip(full_data, enriched_captions):
    original["caption"] = enriched["caption"]

output_path = Path("/Users/timhyde/Desktop/MUSICGEN/text_to_piano_dataset_final.json")
with open(output_path, "w") as f:
    json.dump(full_data, f, indent=2, ensure_ascii=False)

print("Saved")
