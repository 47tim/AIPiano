import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from src.notation.dataset import TextToMIDIDataset
from transformer.transformer_model import MusicTransformer
import os
from miditok import REMI, TokenizerConfig
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import json

# Timothy Hyde 2025
# This file is used to train the old model, the performance based version
# This file is not well commented, it is basically a copy of train.py in src/notation. 
# View that file for full comments.


data_path = Path("data/performance/text_to_piano_dataset_final.json")  #OLD DATASET
filtered_path = Path("data/performance/text_to_piano_dataset_final_filtered.json")

# ---------------------------------------------------------------------------------------------------------------------------------------
# TOKENIZER


tokenizer_config = TokenizerConfig(
    pitch_range=(21, 109),
    
    beat_res={
        (0, 1): 16,
        (1, 2): 8,
        (2, 4): 4,
        (4, 8): 2,
        (8, 16): 1
    },
    beat_res_rest={
        (0, 1): 16,
        (1, 2): 8,
        (2, 4): 4,
        (4, 8): 2,
        (8, 16): 1
    },
    use_rests=True,
    use_chords=True,
    use_positions=True,
    use_bar_embedding=True,
    use_control_changes=True,
    controller_filter=[64],
    use_sustain_pedals=True,
    use_programs=False,
    num_velocities=32,
    default_note_duration=None,
    add_trailing_bars=True
)

midi_tokenizer = REMI(tokenizer_config)
midi_tokenizer.save("config/old_dataset_tokenizer_config")  

# ---------------------------------------------------------------------------------------------------------------------------------------
# CONFIG
BATCH_SIZE = 4 # 4 for low/med
EPOCHS = 500
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = "checkpoints/performance/high" #CHANGE WHEN CHANGING MODEL TRAINING SIZE
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)


cleaned_data = [d for d in raw_data if d.get("midi_tokens") and len(d["midi_tokens"]) > 0]

with open(filtered_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2)

dataset = TextToMIDIDataset(filtered_path)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ---------------------------------------------------------------------------------------------------------------------------------------

# Initializing the model

# LOW MODEL SIZE 
# model = MusicTransformer(
#     text_vocab_size=30000,
#     midi_vocab_size=midi_tokenizer.vocab_size,
#     d_model=512,
#     nhead=8,
#     num_layers=4,
#     dim_feedforward=2048
# ).to(DEVICE)

# MEDIUM MODEL SIZE
# model = MusicTransformer(
#     text_vocab_size=30000,
#     midi_vocab_size=midi_tokenizer.vocab_size,
#     d_model=768,
#     nhead=12,
#     num_layers=6,
#     dim_feedforward=3072
# ).to(DEVICE)

# HIGH MODEL SIZE
model = MusicTransformer(
    text_vocab_size=30000,
    midi_vocab_size=midi_tokenizer.vocab_size,
    d_model=1024,
    nhead=16,
    num_layers=8,
    dim_feedforward=4096
).to(DEVICE)

print("Model initialized on device:", next(model.parameters()).device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = CrossEntropyLoss(ignore_index=0)

# ---------------------------------------------------------------------------------------------------------------------------------------
# TRAINING LOOP

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    # training
    for batch in train_loader:
        text_input, midi_target = batch
        text_input = text_input.to(DEVICE)
        midi_target = midi_target.to(DEVICE)

        midi_input = midi_target[:, :-1]
        midi_expected = midi_target[:, 1:]

        output = model(text_input, midi_input)
        output = output.reshape(-1, output.size(-1))
        midi_expected = midi_expected.reshape(-1)

        loss = loss_fn(output, midi_expected)
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)

    # validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            text_input, midi_target = batch
            text_input = text_input.to(DEVICE)
            midi_target = midi_target.to(DEVICE)

            midi_input = midi_target[:, :-1]
            midi_expected = midi_target[:, 1:]

            output = model(text_input, midi_input)
            output = output.reshape(-1, output.size(-1))
            midi_expected = midi_expected.reshape(-1)

            val_loss = loss_fn(output, midi_expected)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {avg_train_loss:.4f} — Val Loss: {avg_val_loss:.4f}")

    # Saving checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/music_transformer_epoch{epoch+1}.pt")


