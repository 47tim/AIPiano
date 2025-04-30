import json
from pathlib import Path
from miditok import REMI, TokenizerConfig
import tqdm

# Timothy Hyde 2025
# Retokenizing old dataset (text_to_piano_dataset_final.json)

# ---------------------------------------------------------------------------------------------------------------------------------------
# TOKENIZER CONFIG — matches OLD dataset exactly
cfg = TokenizerConfig(
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
    # NOTE: no tempos, no time signatures
)

tk = REMI(cfg)

# ---------------------------------------------------------------------------------------------------------------------------------------
# DATASET PATHS

DATA_JSON = Path("data/performance/text_to_piano_dataset_final.json")    # Old dataset path
MIDI_FOLDER = Path("dataset/performance")                             # MIDI files location

# Load captions + metadata
data = json.load(open(DATA_JSON, encoding="utf-8"))

# Load matching MIDI files
midi_files = sorted(MIDI_FOLDER.glob("*.midi"))

assert len(data) == len(midi_files), f"Mismatch: {len(data)} captions vs {len(midi_files)} MIDIs"

# ---------------------------------------------------------------------------------------------------------------------------------------
# RETOKENIZATION LOOP

for entry, mpath in tqdm.tqdm(zip(data, midi_files), total=len(data), desc="Encoding with old tokenizer settings"):
    try:
        seq = tk.encode(str(mpath))[0].ids   # Convert Path to str
        entry["midi_tokens"] = seq
    except KeyError as e:
        print(f"⚠️ Skipping {mpath} — unknown token: {e}")
        entry["midi_tokens"] = []

# ---------------------------------------------------------------------------------------------------------------------------------------
# SAVE UPDATED JSON

with open(DATA_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("✅ Retokenization complete.")
