from torch.utils.data import DataLoader
from dataset import TextToMIDIDataset

# Timothy Hyde 2025
# This file was just used to test that the dataset was being loaded correctly, and has the correct shape.

dataset = TextToMIDIDataset("C:/Users/school/Desktop/Music Gen Master/MUSICGEN/text_to_piano_dataset_final.json")

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# checking shape here
for batch in dataloader:
    text_inputs, midi_targets = batch
    print("Text:", text_inputs.shape)
    print("MIDI:", midi_targets.shape)  
    break
