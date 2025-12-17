import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import random

class TreeEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, enc_hidden, batch_first=True)
        self.enc_hidden = enc_hidden

    def forward(self, x, lengths):
        """
        x: [B, L] LongTensor (padded fragment indices)
        lengths: [B] LongTensor
        returns: last_hidden: [B, enc_hidden]
        """
        emb = self.embed(x)  # [B, L, E]
        # pack
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.gru(packed)  # h_n: [1, B, enc_hidden]
        return h_n.squeeze(0)  # [B, enc_hidden]
    
class LatentHead(nn.Module):
    def __init__(self, enc_hidden, z_dim):
        super().__init__()
        self.linear_mu = nn.Linear(enc_hidden, z_dim)
        self.linear_logvar = nn.Linear(enc_hidden, z_dim)

    def forward(self, h):
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        return mu, logvar
    
def reparameterize(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std

class TreeDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, dec_hidden, z_dim, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.z_to_hidden = nn.Linear(z_dim, dec_hidden)
        # GRU input: embed_dim + z_dim
        self.gru = nn.GRU(embed_dim + z_dim, dec_hidden, batch_first=True)
        self.out = nn.Linear(dec_hidden, vocab_size)

    def init_hidden_from_z(self, z):
        # z: [B, z_dim] -> [1, B, dec_hidden]
        return torch.tanh(self.z_to_hidden(z)).unsqueeze(0)

    def forward(self, inputs, z, hidden=None):
        # Full sequence forward (teacher forcing)
        B, L = inputs.size()
        if hidden is None:
            hidden = self.init_hidden_from_z(z)

        emb = self.embed(inputs)                               # [B,L,E]
        z_exp = z.unsqueeze(1).expand(-1, L, -1)         # [B,L,z_dim]
        gru_input = torch.cat([emb, z_exp], dim=-1)      # [B,L,E+z]
        out, hidden = self.gru(gru_input, hidden)        # [B,L,H]
        logits = self.out(out)                            # [B,L,vocab]
        return logits, hidden

    def step(self, input_token, z, hidden=None):
        """
        Single time-step for autoregressive inference.
        input_token: [B] long
        z: [B, z_dim]
        hidden: previous hidden
        returns logits [B, vocab], new_hidden
        """
        emb = self.embed(input_token).unsqueeze(1)  # [B,1,E]
        gru_in = torch.cat([emb, z.unsqueeze(1)], dim=-1)  # [B,1,E+z]
        out, h_n = self.gru(gru_in, hidden)  # out: [B,1,dec_hidden]
        logits = self.out(out.squeeze(1))  # [B, vocab]
        return logits, h_n
    
class TreeVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden, dec_hidden, z_dim, pad_idx=0):
        super().__init__()
        self.encoder = TreeEncoder(vocab_size, embed_dim, enc_hidden, pad_idx)
        self.latent = LatentHead(enc_hidden, z_dim)
        self.decoder = TreeDecoder(vocab_size, embed_dim, dec_hidden, z_dim, pad_idx)

    def forward(self, x, lengths, tf_prob=1.0):
        """
        x: [B, L] padded target sequences (we will use teacher forcing)
        lengths: [B] actual lengths
        tf_prob: teacher forcing probability (0..1)
        Returns: logits [B, L, V], mu, logvar
        """
        h_enc = self.encoder(x, lengths)                    # [B, enc_hidden]
        mu, logvar = self.latent(h_enc)                     # [B, z_dim]
        z = reparameterize(mu, logvar).to(device)           # [B, z_dim]

        # Decoder with teacher forcing:
        # For teacher forcing we input the target sequence as inputs (shifted if you want SOS)
        _, hidden = self.decoder(x, z)                       # predict tokens given the inputs (simpler)
        return mu, logvar, z, hidden

    def sample_from_z(self, z, sos_idx=None, max_len=32):
        """
        z: [B, z_dim] latent
        returns: generated indices list (B x <=max_len)
        """
        if sos_idx is None:
            sos_idx = random.choice([17, 9, 5, 11, 2]) #if not specified use top 5 most frequent starting fragments
        self.eval()
        B = z.size(0)
        with torch.no_grad():
            
            # start token: we will assume user uses a special sos index; if none, use first fragment in vocab
            input_tok = torch.full((B,), sos_idx, dtype=torch.long, device=z.device)  # [B]
            hidden = None
            generated = [input_tok.unsqueeze(1)]
            for t in range(max_len):
                logits, hidden = self.decoder.step(input_tok, z, hidden)  # [B, vocab]
                probs = F.softmax(logits, dim=-1)
                # sample or argmax; use sampling to get diverse outputs
                input_tok = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
                generated.append(input_tok.unsqueeze(1))
            gen = torch.cat(generated, dim=1)  # [B, max_len]
        return gen  # indices
