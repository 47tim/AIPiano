import json

# Timothy Hyde 2025
# This file was used to extract only the captions from my custom dataset. This made it easier to work with and add the descriptions
# to each of the pieces, as the file with all the MIDI information is very large and cumbersome to work with.

with open("/Users/timhyde/Desktop/MUSICGEN/text_to_piano_dataset_compact.json", "r") as f:
    full_data = json.load(f)

# Pulling out the captions
captions_only = [{"caption": entry["caption"]} for entry in full_data]

with open("/Users/timhyde/Desktop/MUSICGEN/captions_only.json", "w") as f:
    json.dump(captions_only, f, indent=2)

print("Saved")
