import json
import torch
from transformers import AutoTokenizer
from transformer.transformer_model import MusicTransformer
from dataset import TextToMIDIDataset
from miditok import REMI, TokenizerConfig
from pathlib import Path
import argparse
import miditoolkit
from miditoolkit import MidiFile, Instrument, Note
import pretty_midi
import muspy
import torch.nn.functional as F

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



 # Loading the dataset. "text_to_piano_dataset_final.json"
dataset = TextToMIDIDataset("C:/Users/school/Desktop/Music Gen Master/MUSICGEN/text_to_piano_dataset_final.json")

# cCreating the seed for the MIDI file
midi = MidiFile(ticks_per_beat=480)
instrument = Instrument(program=0, is_drum=False)

# Note DURATION is defined here. 0.5 beat is equal to 240 ticks. 
instrument.notes.append(Note(velocity=100, pitch=60, start=0, end=240))
midi.instruments.append(instrument)
midi.dump(Path("MUSICGEN/seed_tokenizer.mid"))
print("Midi seed saved")

# Verifying prettyMIDI can read the seed MIDI file. Can remove this as it seems to work everytime. 
seed_path = Path(r"C:\Users\school\Desktop\Music Gen Master\MUSICGEN\seed_tokenizer.mid")
try:
    pm = pretty_midi.PrettyMIDI(str(seed_path))
    print("PrettyMIDI success")
except Exception as e:
    raise RuntimeError(f"PrettyMIDI fail: Can't read the seed file! {e}")


example = dataset[0]
_, midi_tokens = example


text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Loading my REMI tokenizer from the json file
with open("MUSICGEN/tokenizer_config/tokenizer.json", "r") as f:
    params = json.load(f)
config_dict = params["config"]

# ------------------------------------------------------------------------------------------------------------------------
# CONVERTING TO TUPLES

# Converting pitch_range into a tuple so it can be read
if "pitch_range" in config_dict:
    config_dict["pitch_range"] = tuple(config_dict["pitch_range"])

# Converting beat_res from strings also into tuples. 
raw_beat_res = config_dict.get("beat_res", {})
converted_beat_res = {tuple(map(int, k.split("_"))): v for k, v in raw_beat_res.items()}
config_dict["beat_res"] = converted_beat_res

# If beat_res_rest exists convert its keys as well
# Checking for beat_res_rest, and converting to tuples as well. Might not be necessary.
# I should check and see if removing this has any effect.
if "beat_res_rest" in config_dict:
    raw_beat_res_rest = config_dict["beat_res_rest"]
    converted_beat_res_rest = {}
    for k, v in raw_beat_res_rest.items():
        if "_" in k:
            converted_beat_res_rest[tuple(map(int, k.split("_")))] = v
        else:
            converted_beat_res_rest[int(k)] = v
    config_dict["beat_res_rest"] = converted_beat_res_rest

# Convert time_signature_range keys: if key is digit string, convert to int;
# if it contains an underscore, convert to tuple; also convert any list values to tuples.
if "time_signature_range" in config_dict:
    raw_ts_range = config_dict["time_signature_range"]
    converted_ts_range = {}
    for k, v in raw_ts_range.items():
        if k.isdigit():
            new_key = int(k)
        elif "_" in k:
            new_key = tuple(map(int, k.split("_")))
        else:
            new_key = k
        if isinstance(v, list):
            v = tuple(v)
        converted_ts_range[new_key] = v
    config_dict["time_signature_range"] = converted_ts_range


# ------------------------------------------------------------------------------------------------------------------------

# Initializing the tokenizer
config = TokenizerConfig(**config_dict)
midi_tokenizer = REMI(config)

seed_tokens = midi_tokenizer.encode(str(seed_path))[0].ids
generated = seed_tokens[:10]  # Use first 10 tokens as the seed

print("Tokenizer vocab size:", midi_tokenizer.vocab_size)

#  parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True, help="Text description of the music")
parser.add_argument("--checkpoint", type=str, default="checkpoints/music_transformer_epoch150.pt") # CHANGE THIS TO EPOCH YOU WANT
parser.add_argument("--output", type=str, default="generated_output.mid")
parser.add_argument("--max_length", type=int, default=2048) # CHANGE THIS To 2048 AFTER TRAINING AGAIN
parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 = disabled)") # TOP K
parser.add_argument("--top_p", type=float, default=0.0, help="Top-p (nucleus) sampling (0.0 = disabled)") # TOP P
args = parser.parse_args()

# Loading the transformer model
TEXT_VOCAB_SIZE = 30000 # Yes I hardcoded this. Sue me. 
MIDI_VOCAB_SIZE = midi_tokenizer.vocab_size # I didn't hardcode this one. Or did I?
print("Model MIDI vocab size: ", MIDI_VOCAB_SIZE)
model = MusicTransformer(
    text_vocab_size=TEXT_VOCAB_SIZE,
    midi_vocab_size=MIDI_VOCAB_SIZE,
    d_model=512,
    nhead=8,
    num_layers=4,
    dim_feedforward=2048
)
model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
model.eval()

# Generating the MIDI tolens using the text prompt
with torch.no_grad():
    encoded = text_tokenizer(
        args.prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64
    )
    text_input = encoded["input_ids"]
    for _ in range(args.max_length):
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

        # Apply top-p (nucleus sampling)
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
        
        generated.append(next_token)
        if next_token == 2:  # stopping at EOS token
            break
        
print("Generated tokens:", generated)

# ------------------------------------------------------------------------------------------------------------------------
# DECODING TOKENS AND SAVING OUTPUT
# I gotta hunch this might be the section that's giving me the squished results. 

# filtering
generated = [t for t in generated if t < midi_tokenizer.vocab_size]
print("Filtered token count:", len(generated))
midi_out = midi_tokenizer.decode([generated])

filtered_tokens = [t for t in generated if t < midi_tokenizer.vocab_size]
print("Filtered token count:", len(filtered_tokens))




score_tick = midi_tokenizer.decode([filtered_tokens]) 

# Manually creating the miditoolkit.MidiFile.
# There is probably a better way to do this
import miditoolkit

# Making Empty midi file
midi_file = miditoolkit.midi.parser.MidiFile(ticks_per_beat=480)

# Creating default tempo. 
# !! TRY LOWERING THIS TO PREVENT SQUISH !! 
midi_file.tempo_changes.append(miditoolkit.TempoChange(tempo=20, time=0))


for track_tick in score_tick.tracks:
   
    # No drums allowed here
    program = getattr(track_tick, "program", 0) or 0
    is_drum = getattr(track_tick, "is_drum", False)

    instrument = miditoolkit.Instrument(program=program, is_drum=is_drum)
    
    # Adding in the notes
    for note_event in track_tick.notes:
        start_tick = note_event.start 
        end_tick = note_event.end
        pitch = note_event.pitch
        velocity = note_event.velocity
        
        note = miditoolkit.Note(
            velocity=velocity,
            pitch=pitch,
            start=start_tick,
            end=end_tick
        )
        instrument.notes.append(note)
    
    midi_file.instruments.append(instrument)

# Stretching notes as a band-aid fix to the squish problem.

#for instrument in midi_file.instruments:
#   instrument.notes.sort(key=lambda n: n.start)

#stretch_factor = 32

#for tempo in midi_file.tempo_changes:
#    tempo.time *= stretch_factor

#for instrument in midi_file.instruments:
#    for note in instrument.notes:
#        note.start = int(note.start * stretch_factor)
#        note.end = int(note.end * stretch_factor)


# Exporting
midi_file.dump(args.output)
print(f"Saved MIDI to: {args.output}")


#midi_out.dump(args.output)
#print(f"Saved MIDI to {args.output}")
