import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import ast

# Timothy Hyde 2025
# This file is for loading the text and MIDI toklen pairs from the text_to_piano_dataset_final.json file
# to be used for model training

class TextToMIDIDataset(Dataset):
    def __init__(self, json_path, text_tokenizer_name="bert-base-uncased", max_text_len=64, max_midi_len=512):
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)
        
        self.max_text_len = max_text_len
        self.max_midi_len = max_midi_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Turning the captions into tokens.
        text_enc = self.text_tokenizer(
            entry["caption"],
            truncation=True,
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt"
        )
        input_ids = text_enc["input_ids"].squeeze(0)  

        if isinstance(entry["midi_tokens"], str):
            midi_tokens = ast.literal_eval(entry["midi_tokens"])  # safely convert string to list
        else:
            midi_tokens = entry["midi_tokens"]

        # Here I'm making sure the length of the MIDI sequence is exactly max_midi_len by either trimming or adding extra 0's.
        midi_tensor = torch.tensor(midi_tokens[:self.max_midi_len], dtype=torch.long)
        if len(midi_tensor) < self.max_midi_len:
            pad_len = self.max_midi_len - len(midi_tensor)
            midi_tensor = torch.cat([midi_tensor, torch.full((pad_len,), 0)])

        return input_ids, midi_tensor
