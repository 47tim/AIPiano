import json
import torch
from transformers import AutoTokenizer
from transformer.transformer_model import MusicTransformer
from src.notation.dataset import TextToMIDIDataset
from miditok import REMI, TokenizerConfig
from pathlib import Path
import argparse
import miditoolkit
from miditoolkit import MidiFile, Instrument, Note, TempoChange
import pretty_midi
import muspy
import torch.nn.functional as F
import random
from tqdm import tqdm

   
# Timothy Hyde 2025
#
# This file is for generating the MIDI sequences, and converting them and saving them in MIDI format.
#
# ------------------------------------------------------------------------------------------------------------------------
# TO-DO LIST:
# 
# 1.  MIDI generations are extremely squished. I'm able to extend these using FL Studio, but for the final application
#       I need to be able to stretch them. Not sure if the squishiness is a result of the actual generation, or if 
#       I'm accidentally squishing them when converting from MIDI tokens back into actual MIDI format. I'm not confident
#       in how I'm handling that conversion. To fix this I can try changing the default tempo, or maybe the single note
#       duration which is set at 0.5 beats. I can also try lowering the default tempo when saving the MIDI file.
# 2. Add Top-K sampling to see if it helps with creating some less predictable, more musical generations. I can also
#       experiment with adding Top-P generation and compare.
# 3. Create the GUI program, and add in a MIDI player so users can play their generations directly from the program
#       instead of having to export and play it that way. More user-friendly, especially for non-musicians.
# 4. Experiment with other ways to better refine the model. I'm also considering extending the length of generations
#       my first attempt was with 512 max_duration, and then I changed it to 2048.
# 5. Add in path variables instead of just hardcoding to my machines path. Sorry I'm lazy.
# 6. Fix issue with timing where all the notes hit at once at the end of the generation
# 7. Move seed tokenizer generator to a seperate file so I don't run it every time I train the model. Can't train and
#       generate at the same time as a result.
# 8. Rename retokenize_with_rests to retokenize_dataset, since I'm using it for all retokenizing of the dataset now.
# 9. Add printing of random seeds, change so tokens are saved in a txt file, not in console.
#
# ------------------------------------------------------------------------------------------------------------------------
# NOTES:
#
# When changing the MAX_LENGTH, you have to change it in 3 places. 1. transformer_model.py line 26; 2. generate.py line 142; 
#
# ------------------------------------------------------------------------------------------------------------------------
# USE:
#
# Generation command format: 
#
# python MUSICGEN/generate.py --prompt "Dreamy, elegant song by Chopin" --checkpoint checkpoints/music_transformer_epoch80.pt --output test4.midi   
#
# ------------------------------------------------------------------------------------------------------------------------

json_path = Path("data/notation/all_captions_cleaned_filtered.json")
seed_path = Path("data/notation/seed_tokenizer.mid")
tokenizer_path = Path("config/notation/tokenizer_config")


dataset = TextToMIDIDataset(json_path)

# creating MIDI file seed
midi = MidiFile(ticks_per_beat=480)
instrument = Instrument(program=0, is_drum=False)

# Note DURATION is defined here. 0.5 beat is equal to 240 ticks. 
instrument.notes.append(Note(velocity=100, pitch=60, start=0, end=240))
midi.instruments.append(instrument)
midi.dump(seed_path)
print("Midi seed saved")

# Verifying prettyMIDI can read the seed MIDI file. Can remove this as it seems to work everytime. 
try:
    pm = pretty_midi.PrettyMIDI(str(seed_path))
    print("PrettyMIDI success")
except Exception as e:
    raise RuntimeError(f"PrettyMIDI fail: Can't read the seed file! {e}")


# This code was for rebuilding and instantiating the tokenizer. No longer necessary, 
# replaced with simple: text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
"""
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

with open("MUSICGEN/tokenizer_config/tokenizer.json", encoding="utf-8") as f:
    params   = json.load(f)
cfg_dict = params["config"]

# cfg_dict["use_time_signatures"] = False

ap = cfg_dict.get("additional_params", {})
if isinstance(ap.get("additional_params"), dict):
    ap = ap["additional_params"]
    cfg_dict["additional_params"] = ap

val = ap.get("max_bar_embedding")
if val is None:
    ap["max_bar_embedding"] = 0
else:
    try:
        ap["max_bar_embedding"] = int(val)
    except (TypeError, ValueError):
        ap["max_bar_embedding"] = 0

# tuple conversions
cfg_dict["beat_res"]      = {tuple(map(int, k.split("_"))): v for k, v in cfg_dict["beat_res"].items()}
cfg_dict["beat_res_rest"] = {tuple(map(int, k.split("_"))): v for k, v in cfg_dict["beat_res_rest"].items()}
if "time_signature_range" in cfg_dict:
    ts = {}
    for k, v in cfg_dict["time_signature_range"].items():
        key = int(k) if k.isdigit() else tuple(map(int, k.split("_")))
        ts[key] = tuple(v)
    cfg_dict["time_signature_range"] = ts

tokenizer_cfg   = TokenizerConfig(**cfg_dict)
midi_tokenizer  = REMI(tokenizer_cfg)
"""

text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
midi_tokenizer = REMI.from_pretrained(tokenizer_path)

# ---------------------------------------------------------------------------------------------------------------------------------------
# SEEDING


# GENERATE SEED USING FIXED TOKENS FROM DATASET MIDI TRACK 0
"""
#example = dataset[0]
#_, midi_tokens = example
#generated = midi_tokens[:32].to list()  
"""

#GENERATE SEED USING RANDOM TOKENS FROM A RANDOM MIDI FROM DATASET
"""
#_, midi_tokens = random.choice(dataset)
#seed_tokens = midi_tokens[:32].tolist()  # Choose up to 64 tokens as seed
#generated = seed_tokens.copy()
"""

# USE CUSTOM SEED

tempo_tokens = [token for token in midi_tokenizer.vocab.keys() if token.startswith("Tempo_")]

generated = [
    midi_tokenizer["BOS_None"],
    midi_tokenizer["Bar_None"],
    midi_tokenizer["Position_0"],
    #midi_tokenizer["Pitch_60"],
    #midi_tokenizer["Velocity_63"],
    #midi_tokenizer["Duration_2.0.4"]   
]
seed_tokens = generated.copy()
print(f"Custom seed initialized with {len(seed_tokens)} tokens.")

# ---------------------------------------------------------------------------------------------------------------------------------------
# PARSING AND LOADING TRANSFORMER MODEL

print("Tokenizer vocab size:", midi_tokenizer.vocab_size)


# Parsing the arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True, help="Text description of the music")
parser.add_argument("--checkpoint", type=str, default="checkpoints/music_transformer_epoch20.pt") # CHANGE THIS TO EPOCH YOU WANT
parser.add_argument("--output", type=str, default="generated_output.mid")
parser.add_argument("--max_length", type=int, default=2048) # CHANGE THIS To 2048 AFTER TRAINING AGAIN
parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 = disabled)") # TOP K
parser.add_argument("--top_p", type=float, default=0.0, help="Top-p (nucleus) sampling (0.0 = disabled)") # TOP P
args = parser.parse_args()



# Loading the transformer model
TEXT_VOCAB_SIZE = 30000 
MIDI_VOCAB_SIZE = midi_tokenizer.vocab_size
print("Model MIDI vocab size: ", MIDI_VOCAB_SIZE)
model = MusicTransformer(
    text_vocab_size=TEXT_VOCAB_SIZE,
    midi_vocab_size=MIDI_VOCAB_SIZE,
    d_model=768,
    nhead=12,
    num_layers=6,
    dim_feedforward=3072     
)
model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
model.eval()
"""

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
    
    for _ in tqdm(range(args.max_length), desc="Generating", ncols=80):
        midi_input = torch.tensor([generated], dtype=torch.long)
        output = model(text_input, midi_input)
        #
        # next_token = torch.argmax(output[0, -1]).item()
        #
        
        logits = output[0, -1]  # shape: [vocab_size]
        probs = F.softmax(logits, dim=-1)

        # Apply top-k
        if args.top_k > 0:
            topk_probs, topk_indices = torch.topk(probs, k=args.top_k)
            probs = torch.zeros_like(probs).scatter(0, topk_indices, topk_probs)

        # Apply top-p 
        if args.top_p > 0.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            cutoff = cumulative_probs > args.top_p
            if torch.any(cutoff):
                last_valid_index = torch.where(cutoff)[0][0] + 1
                sorted_probs[last_valid_index:] = 0.0
                probs = torch.zeros_like(probs).scatter(0, sorted_indices, sorted_probs)

        # Normalize and sample
        probs = probs / probs.sum()  # Make sure total prob = 1
        next_token = torch.multinomial(probs, 1).item()
        
        eos_id = midi_tokenizer["EOS_None"]
        
        generated.append(next_token)
        
        # These lines were creating the issue with note bunching at the end of the generation.
        #if next_token == eos_id:
        #    break
        
    
print("Generated tokens:", generated)

# ------------------------------------------------------------------------------------------------------------------------
# DECODING TOKENS AND SAVING OUTPUT


generated_tokens = generated[len(seed_tokens):]
generated_tokens = [t for t in generated_tokens if t < midi_tokenizer.vocab_size]
print("Filtered token count:", len(generated_tokens))



midi_out = midi_tokenizer.decode([generated_tokens])


id_to_token = {v: k for k, v in midi_tokenizer.vocab.items()}
print("Generated token strings:")
print([id_to_token[token] for token in generated_tokens if token in id_to_token])

filtered_tokens = [t for t in generated_tokens if t < midi_tokenizer.vocab_size]
print("Filtered token count:", len(filtered_tokens))

midi = midi_tokenizer.tokens_to_midi([filtered_tokens])
midi.dump_midi(args.output)
print(f"Saved MIDI to: {args.output}")
