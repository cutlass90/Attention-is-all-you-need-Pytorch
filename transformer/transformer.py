import torch
from torch import nn
from math import sqrt
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


def scaled_dot_product_attention(k, v, q, masked=False):
    # k, v has shape b, steps1, dim
    # q has shape b, step2, dim
    att = torch.bmm(q/sqrt(k.size(2)), k.permute(0,2,1)) # b, step2, step1
    if masked:
        mask = torch.tril(torch.ones(att.shape[1:])).to(att.device)
        att = att.masked_fill(mask == 0, -1e9)
    att = torch.softmax(att, dim=2)
    res = torch.bmm(att, v) # b, step2, dim
    return res

class MultiHeadAttention(nn.Module):
    def __init__(self, h=8, d_model=64, d_k=64, d_q=64, masked=False):
        """
        Args:
            h: int, number of heads
            d_model: int, dimetion of input embeddings
        """
        super().__init__()
        self.h = h
        self.masked = masked
        self.linear_k, self.linear_v, self.linear_q = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(h):
            self.linear_k.append(nn.Linear(d_model, d_k))
            self.linear_v.append(nn.Linear(d_model, d_k))
            self.linear_q.append(nn.Linear(d_model, d_q))
        self.final_liner = nn.Linear(h*d_k, d_model)

    def forward(self, k, v, q):
        res = []
        for i in range(self.h):
            a = scaled_dot_product_attention(k=self.linear_k[i](k),
                                             v=self.linear_v[i](v),
                                             q=self.linear_q[i](q),
                                             masked=self.masked)
            res.append(a)
        concated = torch.cat(res, dim=2)
        return self.final_liner(concated)


class Encoder(nn.Module):
    def __init__(self, h=8, d_model=64, d_k=64, d_q=64, N=6):
        """
        Args:
            h: int, number of heads
            d_model: int, dimetion of input embeddings
            N: int, number of blocks
        """
        super().__init__()
        self.N = N
        self.net = nn.ModuleDict()
        for i in range(N):
            self.net[f'mha{i}'] = MultiHeadAttention(h, d_model, d_k, d_q)
            self.net[f'linear{i}'] = nn.Sequential(
                nn.Linear(d_model, d_model*4),
                nn.ReLU(),
                nn.Linear(d_model*4, d_model)
                )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input_sequence):
        self.layer_norm(input_sequence)
        for i in range(self.N):
            input_sequence = self.layer_norm(input_sequence + self.net[f'mha{i}'](input_sequence, input_sequence, input_sequence))
            input_sequence = self.layer_norm(input_sequence + torch.relu(self.net[f'linear{i}'](input_sequence)))
        return input_sequence



class Decoder(nn.Module):
    def __init__(self, h=8, d_model=64, d_k=64, d_q=64, N=6):
        """
        Args:
            h: int, number of heads
            d_model: int, dimetion of input embeddings
            N: int, number of blocks
        """
        super().__init__()
        self.N = N
        self.net = nn.ModuleDict()
        for i in range(N):
            self.net[f'mmha{i}'] = MultiHeadAttention(h, d_model, d_k, d_q, masked=True)
            self.net[f'mha{i}'] = MultiHeadAttention(h, d_model, d_k, d_q)
            self.net[f'linear{i}'] = nn.Sequential(
                nn.Linear(d_model, d_model*4),
                nn.ReLU(),
                nn.Linear(d_model*4, d_model)
                )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, encoded_sequence, output_sequence):
        output_sequence = self.layer_norm(output_sequence)
        for i in range(self.N):
            output_sequence = self.layer_norm(output_sequence + self.net[f'mmha{i}'](output_sequence, output_sequence, output_sequence))
            output_sequence = self.layer_norm(output_sequence + self.net[f'mha{i}'](encoded_sequence, encoded_sequence, output_sequence))
            output_sequence = self.layer_norm(output_sequence + self.net[f'linear{i}'](output_sequence))
        return output_sequence

class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, h=8, d_model=512, d_k=64, d_q=64, N=6, n_position=200, emb_src_trg_weight_sharing=True):
        super().__init__()
        self.linear_logits = nn.Linear(d_model, n_trg_vocab)
        self.encoder = Encoder(h, d_model, d_k, d_q, N)
        self.decoder = Decoder(h, d_model, d_k, d_q, N)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=src_pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        if emb_src_trg_weight_sharing:
            self.trg_word_emb = self.src_word_emb
        else:
            self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=trg_pad_idx)
    def forward(self, input_seq, output_seq):
        input_seq = self.src_word_emb(input_seq)
        input_seq = self.position_enc(input_seq)
        encoded_seq = self.encoder(input_seq)

        output_seq = self.trg_word_emb(output_seq)
        output_seq = self.position_enc(output_seq)
        output_seq = self.decoder(encoded_seq, output_seq)
        logits = torch.softmax(self.linear_logits(output_seq), dim=2)
        return logits.view(-1, logits.size(2))





if __name__ == "__main__":
    # mha = MultiHeadAttention()
    # k, v, q = torch.randn(2, 4, 64), torch.randn(2, 4, 64), torch.randn(2, 4, 64)
    # res = mha(k, v, q)
    # print()

    # encoder = Encoder()
    # input_sequence = torch.randn(2,4,64)
    # output = encoder(input_sequence)
    # print()

    # mha = MultiHeadAttention(masked=True)
    # k, v, q = torch.randn(2, 4, 64), torch.randn(2, 4, 64), torch.randn(2, 4, 64)
    # res = mha(k, v, q)
    # print()

    # decoder = Decoder(1000)
    # encoded_sequence, output_sequence = torch.randn(2, 4, 64), torch.randn(2, 5, 64)
    # logits = decoder(encoded_sequence, output_sequence)
    # print()

    transformer = Transformer(n_src_vocab=10000, n_trg_vocab=20000, src_pad_idx=100, trg_pad_idx=100)
    input_sequence, output_sequence = torch.randint(1000, (2, 4)).long(), torch.randint(1000, (2, 5)).long()
    res = transformer(input_sequence, output_sequence)


