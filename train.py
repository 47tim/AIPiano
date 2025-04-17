import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from dataset import TextToMIDIDataset
from transformer.transformer_model import MusicTransformer
import os
from miditok import REMI, TokenizerConfig
from pathlib import Path

# Timothy Hyde 2025
# This is the file used for training the model

# This has to match generate.py EXACTLY!!!!
tokenizer_config = TokenizerConfig(
    pitch_range=(21, 109),
    use_chords=False,
    use_programs=False,
    use_tempos=True,
    use_time_signatures=False,
    num_velocities=32,
    beat_res={(0, 4): 8, (4, 12): 4},
)

midi_tokenizer = REMI(tokenizer_config)

midi_tokenizer.save_params("MUSICGEN/tokenizer_config/")


# config
BATCH_SIZE = 4
EPOCHS = 150
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = "checkpoints"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Loading the dataset
dataset = TextToMIDIDataset("C:/Users/school/Desktop/Music Gen Master/MUSICGEN/text_to_piano_dataset_final.json")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initializing the model
model = MusicTransformer(
    text_vocab_size=30000,  # tokenizer size
    midi_vocab_size=314,    # from miditok vocab
    d_model=512,
    nhead=8,
    num_layers=4,
    dim_feedforward=2048
).to(DEVICE)

# Checking to make sure its using my GPU. Can remove
print("Model on device:", next(model.parameters()).device)


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = CrossEntropyLoss(ignore_index=0)  # ignore padding token

# ------------------------------------------------------------------------------------------------------------------------
# TRAINING LOOP
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in dataloader:
        text_input, midi_target = batch
        text_input = text_input.to(DEVICE)
        midi_target = midi_target.to(DEVICE)

        # Shift targets
        midi_input = midi_target[:, :-1]
        midi_expected = midi_target[:, 1:]

        # Forward pass
        output = model(text_input, midi_input)  # shape: [B, T, V]
        output = output.reshape(-1, output.size(-1))       # [B*T, V]
        midi_expected = midi_expected.reshape(-1)          # [B*T]

        loss = loss_fn(output, midi_expected)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

    # Saving checkpoint after each epoch
    torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/music_transformer_epoch{epoch+1}.pt")
