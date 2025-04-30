import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import TextToMIDIDataset
from miditok import REMI

# Timothy Hyde 2025
# This script was used to scan my dataset and see if any tempo tokens were appearing,
# since my generations were not producing any tempo tokens.

dataset = TextToMIDIDataset("C:/Users/school/Desktop/Music Gen Master/MUSICGEN/new_text_to_piano_dataset_final.json")
midi_tokenizer = REMI.from_pretrained("MUSICGEN/tokenizer_config/")
id_to_token = {v: k for k, v in midi_tokenizer.vocab.items()}
NUM_SAMPLES = 10

for idx in range(NUM_SAMPLES):
    
    print(f"\n--- Sample {idx} ---")
    text_input, midi_tokens = dataset[idx]

    token_strings = [id_to_token.get(token.item(), "UNK") for token in midi_tokens if token.item() in id_to_token]

    has_tempo = any(tok.startswith("Tempo_") for tok in token_strings)
    has_pedal = any(tok.startswith("Pedal_") or tok.startswith("PedalOff_") for tok in token_strings)

    print(f"Tempo token found: {has_tempo}")
    print(f"Pedal token found: {has_pedal}")

    print("First 20 tokens:", token_strings[:20])
