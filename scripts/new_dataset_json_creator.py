import os
import json

# Timothy Hyde 2025
# This file was for creating my new datset JSON file


def format_caption(composer, title):
    composer = composer.capitalize()
    title = title.replace('_', ' ').capitalize()

    return f"'{title}' by {composer}"

def create_caption_json(midi_folder, output_json):
    captions = []

    midi_files = sorted(f for f in os.listdir(midi_folder) if f.endswith(".mid"))

    for file in midi_files:
        try:
            parts = file[:-4].split("_", 2) 
            if len(parts) < 3:
                print(f"Skipping filename: {file}")
                continue

            _, composer, title = parts
            caption = format_caption(composer, title)
            captions.append({"caption": caption})
        except Exception as e:
            print(f"Error processing {file}: {e}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2)
    print(f"Saved {len(captions)} captions to {output_json}")


create_caption_json(
    midi_folder=r"C:\Users\school\Desktop\Music Gen Master\DATA",
    output_json="new_text_to_piano_dataset.json"
)
