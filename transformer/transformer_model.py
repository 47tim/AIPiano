import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Timothy Hyde 2025
# This is the transformer model file

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * -(math.log(10000.0) / emb_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, emb_size]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)
    
    
    
class MusicTransformer(nn.Module):
    def __init__(self, text_vocab_size, midi_vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, max_len=2048): #CHANGE THIS MAX_LEN
        super().__init__()
        self.text_embed = nn.Embedding(text_vocab_size, d_model)
        # Had to add the plus one...
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len + 50)



        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.target_embed = nn.Embedding(midi_vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, midi_vocab_size)

    def forward(self, text_input, midi_input):
        # text_input: [batch, seq_len] (captions)
        # midi_input: [batch, seq_len] (tokens generated so far)

        text_emb = self.pos_encoder(self.text_embed(text_input))  # [batch, seq_len, d_model]
        midi_emb = self.pos_encoder(self.target_embed(midi_input))

        text_emb = text_emb.permute(1, 0, 2)
        midi_emb = midi_emb.permute(1, 0, 2)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(midi_emb.size(0)).to(midi_emb.device)

        output = self.decoder(midi_emb, text_emb, tgt_mask=tgt_mask)  # [seq_len, batch, d_model]
        output = self.fc_out(output)  # [seq_len, batch, midi_vocab_size]

        return output.permute(1, 0, 2)  # [batch, seq_len, vocab]

