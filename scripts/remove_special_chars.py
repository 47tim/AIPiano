import json
import unicodedata
from pathlib import Path

# Timothy Hyde 2025
# This script was used to remove any special chars from my dataset captions


input_path = Path(r"C:\Users\school\Desktop\Music Gen Master\MUSICGEN\all_captions_datset.json")
output_path = Path(r"C:\Users\school\Desktop\Music Gen Master\MUSICGEN\all_captions_cleaned.json")


with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)


def normalize_text(text):
    normalized = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
    normalized = normalized.replace('"', "").replace("'", "")  
    return normalized

for entry in data:
    if "caption" in entry:
        entry["caption"] = normalize_text(entry["caption"])


with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Cleaned captions saved to: {output_path}")
