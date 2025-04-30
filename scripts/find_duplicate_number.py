import os
import re
from collections import defaultdict
from pathlib import Path

# Timothy Hyde 2025
# This script was used for finding MIDI files with the same number in my dataset.

folder = Path(r"C:\Users\school\Desktop\DATA")
pattern = re.compile(r"^(\d{4})_")
number_map = defaultdict(list)

for file in folder.iterdir():
    if file.is_file():
        match = pattern.match(file.name)
        if match:
            number = match.group(1)
            number_map[number].append(file.name)

duplicates = {num: names for num, names in number_map.items() if len(names) > 1}

if duplicates:
    print("Duplicate numbers found:\n")
    for num, files in sorted(duplicates.items()):
        print(f"{num}:")
        for f in files:
            print(f"  - {f}")
else:
    print("No duplicate numbers found.")
