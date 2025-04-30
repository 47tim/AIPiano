import json
import torch
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer
from miditok import REMI, TokenizerConfig
from transformer.transformer_model import MusicTransformer
from miditoolkit import MidiFile, Instrument, Note
import pretty_midi

# Timothy Hyde 2025
# This file is used to generate MIDI from old dataset model, the performance version
# This file is not well commented, it is basically a copy of generate.py in src/notation. 
# View that file for full comments.

# ---------------------------------------------------------------------------------------------------------------------------------------
# CONFIG:

json_path = Path("data/performance/text_to_piano_dataset_final_filtered.json")
seed_path = Path("data/performance/seed_tokenizer.mid")
tokenizer_path = Path("config/performance/tokenizer.json")


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
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# creating MIDI file seed
midi = MidiFile(ticks_per_beat=480)
instrument = Instrument(program=0, is_drum=False)
instrument.notes.append(Note(velocity=100, pitch=60, start=0, end=240)) 
midi.instruments.append(instrument)
midi.dump(seed_path)
print("Seed MIDI saved")

# Verifying prettyMIDI can read the seed MIDI file. Can remove this as it seems to work everytime. 
try:
    pm = pretty_midi.PrettyMIDI(str(seed_path))
    print("PrettyMIDI read successful")
except Exception as e:
    raise RuntimeError(f"PrettyMIDI read failed: {e}")


generated = [
    midi_tokenizer["BOS_None"],
    midi_tokenizer["Bar_None"],
    midi_tokenizer["Position_0"]
]
seed_tokens = generated.copy()
print(f"Custom seed initialized with {len(seed_tokens)} tokens.")


# Parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True, help="Text description of the music")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint filename (like music_transformer_epoch30.pt)")
parser.add_argument("--output", type=str, default="generated_output.mid")
parser.add_argument("--max_length", type=int, default=2048)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--top_p", type=float, default=0.0)
parser.add_argument("--model_size", type=str, default="medium", help="Choose model size: low, medium, or high")
args = parser.parse_args()

full_checkpoint_path = Path(args.checkpoint)

# Loading the transformer model
TEXT_VOCAB_SIZE = 30000 
MIDI_VOCAB_SIZE = midi_tokenizer.vocab_size
print("Model MIDI vocab size: ", MIDI_VOCAB_SIZE)

if args.model_size == "low":
    model = MusicTransformer(
        text_vocab_size=TEXT_VOCAB_SIZE,
        midi_vocab_size=MIDI_VOCAB_SIZE,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048
    )
elif args.model_size == "medium":
    model = MusicTransformer(
        text_vocab_size=TEXT_VOCAB_SIZE,
        midi_vocab_size=MIDI_VOCAB_SIZE,
        d_model=768,
        nhead=12,
        num_layers=6,
        dim_feedforward=3072
    )
elif args.model_size == "high":
    model = MusicTransformer(
        text_vocab_size=TEXT_VOCAB_SIZE,
        midi_vocab_size=MIDI_VOCAB_SIZE,
        d_model=1024,
        nhead=16,
        num_layers=8,
        dim_feedforward=4096
    )
else:
    raise ValueError("Invalid model size.")

model.load_state_dict(torch.load(full_checkpoint_path, map_location="cpu"))
model.eval()

print(f"Model loaded, MIDI vocab size: {MIDI_VOCAB_SIZE}")

# ---------------------------------------------------------------------------------------------------------------------------------------
# MIDI GENERATION

# Generating the MIDI tokens using the text prompt
with torch.no_grad():
    encoded = text_tokenizer(
        args.prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64
    )
    text_input = encoded["input_ids"]

    for _ in tqdm(range(args.max_length), desc="Generating tokens", ncols=80):
        midi_input = torch.tensor([generated], dtype=torch.long)
        output = model(text_input, midi_input)
        
        logits = output[0, -1] 
        probs = F.softmax(logits, dim=-1)

        # top-k
        if args.top_k > 0:
            topk_probs, topk_indices = torch.topk(probs, k=args.top_k)
            probs = torch.zeros_like(probs).scatter(0, topk_indices, topk_probs)

        # top-p
        if args.top_p > 0.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            cutoff = cumulative_probs > args.top_p
            if torch.any(cutoff):
                last_valid_index = torch.where(cutoff)[0][0] + 1
                sorted_probs[last_valid_index:] = 0.0
                probs = torch.zeros_like(probs).scatter(0, sorted_indices, sorted_probs)

        probs = probs / probs.sum()
        next_token = torch.multinomial(probs, 1).item()

        generated.append(next_token)

print(f"Complete. Total tokens: {len(generated)}")

# ---------------------------------------------------------------------------------------------------------------------------------------
# DECODING TOKENS AND SAVING OUTPUT

generated_tokens = generated[len(seed_tokens):]
filtered_tokens = [t for t in generated_tokens if t < midi_tokenizer.vocab_size]

print(f"Filtered token count: {len(filtered_tokens)}")

midi = midi_tokenizer.tokens_to_midi([filtered_tokens])
midi.dump_midi(args.output)
print(f"MIDI saved to {args.output}")
