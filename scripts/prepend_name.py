from pathlib import Path

# Timothy Hyde 2025
# This script was to append composer names to the start of my manually downloaded MIDI files from KDF


FOLDER = Path(r"C:\Users\school\Desktop\kdf_2")  
PREFIX = "faure_"  


for file in FOLDER.iterdir():
    if file.is_file():
        new_name = PREFIX + file.name
        new_path = file.with_name(new_name)
        file.rename(new_path)

print("Complete")
