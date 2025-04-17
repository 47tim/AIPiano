import torch
from transformer_model import MusicTransformer

# Timothy Hyde 2025
# This file was just used to test the transformer model and make sure it works before moving forward.

# Same sizes I hardcoded into the other files.
TEXT_VOCAB_SIZE = 30000   
MIDI_VOCAB_SIZE = 512     

BATCH_SIZE = 2
TEXT_SEQ_LEN = 32
MIDI_SEQ_LEN = 128

# random
text_input = torch.randint(0, TEXT_VOCAB_SIZE, (BATCH_SIZE, TEXT_SEQ_LEN))
midi_input = torch.randint(0, MIDI_VOCAB_SIZE, (BATCH_SIZE, MIDI_SEQ_LEN))

model = MusicTransformer(
    text_vocab_size=TEXT_VOCAB_SIZE,
    midi_vocab_size=MIDI_VOCAB_SIZE,
    d_model=512,
    nhead=8,
    num_layers=4,
    dim_feedforward=2048
)

output = model(text_input, midi_input)

# Making sure the shape is what I expect
print("{output.shape}")
assert output.shape == (BATCH_SIZE, MIDI_SEQ_LEN, MIDI_VOCAB_SIZE)
