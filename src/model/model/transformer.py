import torch
import torch.nn as nn
import math


def generate_square_subsequent_mask(sz, device="cuda"):
    mask = torch.triu(torch.ones((sz, sz))).to(device) == 1
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt, device="cuda"):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    src_padding_mask = (src == torch.zeros(src.shape[2]).to(src.device))[:, :, 0]
    tgt_padding_mask = (tgt == torch.zeros(tgt.shape[2]).to(src.device))[:, :, 0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).to(device).type(torch.bool)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class Transformer4Input(nn.Module):
    def __init__(
        self,
        dim_model=32,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        resolution=32,
    ):
        super().__init__()

        # Layers
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout=dropout, max_len=5000
        )

        self.output_first_token = nn.Parameter(torch.zeros(8 * 32))

    def forward(
        self,
        x: torch.Tensor,
        src_mask=None,
        tgt_mask=None,
        src_pad_mask=None,
        tgt_pad_mask=None,
    ):
        initial_shape = x.shape
        x = x.reshape((x.size(0), x.size(1), x.size(2) * x.size(3)))
        src = self.positional_encoder(x)
        # tgt = self.positional_encoder(y)
        tgt = self.output_first_token.repeat(100, 1).unsqueeze(1)

        # tgt = torch.cat(
        #     [torch.stack([self.output_first_token] * src.size(0)).unsqueeze(1), tgt],
        #     dim=1,
        # )

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt, device=x.device
        )

        encoded_condition = self.transformer(
            src,
            tgt,
            # src_mask=src_mask,
            # tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            # tgt_key_padding_mask=tgt_padding_mask,
        )
        encoded_condition = encoded_condition.reshape((src.size(0), *initial_shape[2:]))
        return encoded_condition


class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout,
        resolution,
    ):
        super().__init__()

        self.resolution = resolution
        # Layers
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
        )
        self.out = nn.Linear(dim_model, resolution)
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout=dropout, max_len=5000
        )

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return matrix == pad_token

    def forward(self, x, y, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        src = self.positional_encoder(x)
        tgt = self.positional_encoder(y)
        if y is not None:
            tgt = torch.cat([torch.randn_like(y[0].unsqueeze(0)), tgt], axis=0)
        else:
            tgt = torch.randn_like

        self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )

    def inference(self, input_sequence, out_shape):

        self.eval()

        y_input = torch.randn(out_shape)

        for _ in range(input_sequence.size(0)):
            # Get source mask
            tgt_mask = self.get_tgt_mask(y_input.size(1))

            pred = self(input_sequence, y_input, tgt_mask)

            next_item = (
                pred.topk(1)[1].view(-1)[-1].item()
            )  # num with highest probability
            next_item = torch.tensor([[next_item]])

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == EOS_token:
                break

        return y_input.view(-1).tolist()


class Transformer4DDPM(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout,
        resolution,
    ):
        super().__init__()

        self.resolution = resolution
        # Layers
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(dim_model, resolution)
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout=dropout, max_len=5000
        )

    def time_pos_encoding(self, t, channels):
        inv_freq = (
            1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels)).cuda()
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return matrix == pad_token

    def forward(self, x, t, y):
        src = self.positional_encoder(x)
        tgt = self.positional_encoder(y)
        tgt = torch.cat([torch.randn_like(y[0].unsqueeze(0)), tgt], axis=0)

        transformer_out = self.transformer()


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout, max_len):
        super().__init__()
        den = torch.exp(-torch.arange(0, dim_model, 2) * math.log(10000) / dim_model)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, dim_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(
            token_embedding
            + self.pos_embedding[: token_embedding.size(1), :].repeat((100, 1, 1))
        )
