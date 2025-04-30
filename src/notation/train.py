import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from src.notation.dataset import TextToMIDIDataset
from transformer.transformer_model import MusicTransformer
import os
from miditok import REMI, TokenizerConfig
from pathlib import Path
import json

# Timothy Hyde 2025
# This is the file used for training the model


data_path = Path("data/notation/all_captions_cleaned.json")
filtered_path = Path("data/notation/all_captions_cleaned_filtered.json")

# ---------------------------------------------------------------------------------------------------------------------------------------
# TOKENIZER

# This has to match generate.py EXACTLY!!!!
tokenizer_config = TokenizerConfig(
    pitch_range=(21, 109),
    
    #testing 
    #beat_res={(0,16):8},   
    #beat_res={(0, 4): 4, (4, 16): 2},
    beat_res = {
    (0, 1): 16,  
    (1, 2): 8,    
    (2, 4): 4,     
    (4, 8): 2,    
    (8, 16): 1     
    },
     
    beat_res_rest = {
    (0, 1): 16,   
    (1, 2): 8,    
    (2, 4): 4,     
    (4, 8): 2,    
    (8, 16): 1   
    },
    
    use_rests=True,
    use_chords=True,
    use_time_signatures=True,
    use_tempos=True,
    use_positions=True,
    use_bar_embedding=True, #new
    use_control_changes=True, #new
    controller_filter=[64], #new
    use_sustain_pedals=True, #new                         
    use_programs=False,
    num_velocities=32,
    default_note_duration=None,
    add_trailing_bars=True,
)

midi_tokenizer = REMI(tokenizer_config)

midi_tokenizer.save("config/notation/tokenizer_config")

# ---------------------------------------------------------------------------------------------------------------------------------------
# CONFIG

BATCH_SIZE = 4
EPOCHS = 500
LEARNING_RATE = 5e-6 # 1e-4 usual, 1e-5 low, 5e-6 even lower
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = "checkpoints/notation/medium"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------------------------------------------------------------------
# LOADING DATASET AND INITIALIZING MODEL

# Loading the dataset
#dataset = TextToMIDIDataset("C:/Users/school/Desktop/Music Gen Master/MUSICGEN/text_to_piano_dataset_final.json")
#dataset = TextToMIDIDataset("C:/Users/school/Desktop/Music Gen Master/MUSICGEN/new_text_to_piano_dataset_final.json")

# ---------------------------------------------------------------------------------------------------------------------------------------
# Filtering out empty token data (looking at you joplin)

with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Filter out broken or skipped entries
cleaned_data = [d for d in raw_data if d.get("midi_tokens") and len(d["midi_tokens"]) > 0]

# Saving to temporary file

with open(filtered_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2)


dataset = TextToMIDIDataset(filtered_path)

# splitting for training and validation, 90 and 10 percent
train_size = int(0.9 * len(dataset))  
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ---------------------------------------------------------------------------------------------------------------------------------------

# Initializing the model

# LOW
# model = MusicTransformer(
#     text_vocab_size=30000,  # tokenizer size
#     midi_vocab_size=midi_tokenizer.vocab_size,    
#     d_model=512,
#     nhead=8,
#     num_layers=4,
#     dim_feedforward=2048
# ).to(DEVICE)


# MEDIUM
model = MusicTransformer(
    text_vocab_size=30000,
    midi_vocab_size=midi_tokenizer.vocab_size,
    d_model=768,
    nhead=12,
    num_layers=6,
    dim_feedforward=3072
).to(DEVICE)

# If resuming training from an epoch, use this block. If not, comment out.

checkpoint_path = "C:/Users/school/Desktop/Music Gen Master/MUSICGEN/checkpoints/notation/medium/music_transformer_epoch40.pt" 
start_epoch = 50
if os.path.exists(checkpoint_path):
    print(f" Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    start_epoch = int(checkpoint_path.split("epoch")[-1].split(".")[0])  # extracts num epoch from filename


# Checking to make sure its using my GPU. Can remove
print("Model on device:", next(model.parameters()).device)


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = CrossEntropyLoss(ignore_index=0)  # ignore padding token

# ------------------------------------------------------------------------------------------------------------------------
# TRAINING LOOP


#for epoch in range(EPOCHS):
for epoch in range(start_epoch, EPOCHS):
 
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