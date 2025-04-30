import json, glob, tqdm
from miditok import REMI, TokenizerConfig
from pathlib import Path

# RETOKENIZE MAIN FILE

# 1) Rebuild the exact same TokenizerConfig you used for training
cfg = TokenizerConfig(
    pitch_range=(21, 109),
    #beat_res={(0, 16): 8},
    #beat_res={(0, 4): 4, (4, 16): 2},
   beat_res = {
    (0, 1): 16,    # 64th notes
    (1, 2): 8,     # 32nd notes
    (2, 4): 4,     # 16th notes
    (4, 8): 2,     # 8th notes
    (8, 16): 1     # quarters & longer
    },
    beat_res_rest = {
    (0, 1): 16,    # 64th-note resolution
    (1, 2): 8,     # 32nd-note resolution
    (2, 4): 4,     # 16th-note resolution
    (4, 8): 2,     # 8th-note resolution
    (8, 16): 1     # quarter notes and longer
    },
    use_rests=True,
    use_chords=True,
    use_time_signatures=True,
    use_tempos=True,
    use_programs=False,
    use_bar_embedding=True, #new
    use_control_changes=True, #new
    controller_filter=[64], #new
    use_sustain_pedals=True, #new
    num_velocities=32,
    use_positions=True,
    # default_note_duration=4.0, 
    default_note_duration=None,
    add_trailing_bars=True
    #additional_params={
    #    "max_bar_embedding": 0,
    #    "use_bar_end_tokens": False,
    #    "add_trailing_bars": False
    #}
)

tk = REMI(cfg)

# Loading JSON with my captions
DATA_JSON = Path("data/notation/all_captions_cleaned.json")
MIDI_FOLDER = Path("dataset/notation")


data = json.load(open(DATA_JSON, encoding="utf-8"))

midi_files = sorted(MIDI_FOLDER.glob("*.mid"))
assert len(data) == len(midi_files), f"{len(data)} captions vs {len(midi_files)} MIDIs"

# Re encoding
for entry, mpath in tqdm.tqdm(zip(data, midi_files), total=len(data), desc="Encoding with rests"):
    try:
        seq = tk.encode(mpath)[0].ids
        entry["midi_tokens"] = seq
    except KeyError as e:
        print(f" Skipping {mpath} — unknown token: {e}")
        entry["midi_tokens"] = []

# Overwriting JSON
with open(DATA_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Retokenization complete")
