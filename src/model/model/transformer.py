import torch
import torch.nn as nn
import math


def generate_square_subsequent_mask(sz, device="cuda"):
    mask = (
        torch.triu(
            torch.ones((sz, sz)),
        ).to(device)
        == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt, device="cuda"):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    # src_padding_mask = (src == torch.zeros(src.shape[2]).to(src.device))[:, :, 0]
    # tgt_padding_mask = (tgt == torch.zeros(tgt.shape[2]).to(src.device))[:, :, 0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).to(device).type(torch.bool)

    return src_mask, tgt_mask  # , src_padding_mask, tgt_padding_mask


def time_encoding(t, channels):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels)).cuda()
    # repeat t tensor channels//2 times ona second dimension

    pos_enc_a = torch.sin(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
    pos_enc_b = torch.cos(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc


class Transformer4Input(nn.Module):
    def __init__(
        self,
        dim_model=32,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        dim_feed_forward=2048,
        resolution=32,
    ):
        super().__init__()

        # Layers
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.dim_feed_forward = dim_feed_forward

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
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):
        initial_shape = x.shape
        x = x.reshape((x.size(0), x.size(1), x.size(2) * x.size(3)))
        src = self.positional_encoder(x)
        # tgt = self.positional_encoder(y)
        tgt = self.output_first_token.repeat(x.size(0), 1).unsqueeze(1)

        # tgt = torch.cat(
        #     [torch.stack([self.output_first_token] * src.size(0)).unsqueeze(1), tgt],
        #     dim=1,
        # )

        src_mask, tgt_mask = create_mask(x, tgt, device=x.device)

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


class TestModel(nn.Module):
    def __init__(
        self,
        in_channels=8,
        out_channels=5,
        resolution=32,
        dim_model=32,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        dim_feed_forward=2048,
        time_dim=256,
        encoding_layer=None,
        conditional=True,
    ):
        super().__init__()

        # Layers
        self.c_in = in_channels
        self.c_out = out_channels
        self.resolution = resolution
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.dim_feed_forward = dim_feed_forward
        self.time_dim = time_dim
        self.conditional = conditional

        self.model = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
        )

        # self.output_first_token = nn.Parameter(torch.zeros(in_channels * resolution))

    def time_encoding(self, t, channels):
        inv_freq = (
            1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels)).cuda()
        )

        pos_enc_a = torch.sin(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(
        self,
        x: torch.Tensor,
        t,
        y,
        src_padding_mask=None,
    ):
        src = x.squeeze(1)

        t = self.time_encoding(t, self.time_dim)
        src += t

        return self.model(src)


class Transformer4DDPM2(nn.Module):
    def __init__(
        self,
        in_channels=8,
        out_channels=5,
        resolution=32,
        dim_model=32,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        dim_feed_forward=2048,
        time_dim=256,
        encoding_layer=None,
        conditional=True,
    ):
        super().__init__()

        # Layers
        self.c_in = in_channels
        self.c_out = out_channels
        self.resolution = resolution
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.dim_feed_forward = dim_feed_forward
        self.time_dim = time_dim
        self.conditional = conditional

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

        self.post_freq_embedding_time = nn.Sequential(
            nn.Linear(self.time_dim, in_channels * resolution),
            nn.ReLU(),
        )

        self.output_first_token = nn.Parameter(torch.zeros(in_channels * resolution))

    def time_encoding(self, t, channels):
        inv_freq = (
            1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels)).cuda()
        )

        pos_enc_a = torch.sin(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        src = src.reshape((src.size(0), src.size(1), src.size(2) * src.size(3)))
        src = self.positional_encoder(src)

        return self.transformer.encoder(src, src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoder(tgt), memory, tgt_mask)

    def forward(
        self,
        x: torch.Tensor,
        t,
        y,
        src_padding_mask=None,
    ):
        src = x

        t = self.time_encoding(t, self.time_dim)
        t = self.post_freq_embedding_time(t)

        tgt = t.unsqueeze(1)
        # tgt = torch.stack([self.output_first_token] * src.size(0), dim=0).unsqueeze(1)

        # src = torch.cat([t.unsqueeze(1), src], dim=1)
        src = self.positional_encoder(src)
        # tgt = self.positional_encoder(tgt)

        # src += tgt

        src_mask, tgt_mask = create_mask(src, tgt, device=src.device)

        output_sequence = self.transformer(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=None,
            tgt_key_padding_mask=None,
        )

        out = output_sequence.reshape((src.size(0), tgt.size(1), self.resolution))

        return out

    # def generate_out_sequence(self, past_surfaces, observations, t):
    #     src = past_surfaces

    #     len_out_seq = src.size(1)

    #     src_mask = (
    #         torch.zeros((len_out_seq, len_out_seq))
    #         .to(past_surfaces.device)
    #         .type(torch.bool)
    #     )

    #     memory = self.encode(src, src_mask, t)

    #     t = self.time_encoding(t, self.time_dim)
    #     t = self.post_freq_embedding_time(t)

    #     out_sequence = t.unsqueeze(1)

    #     if self.conditional:
    #         out_sequence_converted = out_sequence
    #     else:
    #         out_sequence_converted = out_sequence

    #     for i in range(len_out_seq):
    #         tgt_mask = generate_square_subsequent_mask(out_sequence.size(1))
    #         out = self.decode(out_sequence, memory, tgt_mask)
    #         out_sequence = torch.cat([out_sequence, out[:, -1:, :]], dim=1)
    #         out_sequence_converted = torch.cat(
    #             [out_sequence_converted, out[:, -1:, :]], dim=1
    #         )

    #     return out_sequence_converted.reshape(
    #         (
    #             out_sequence.size(0),
    #             out_sequence.size(1),
    #             src.size(2),
    #             self.resolution,
    #         )
    #     )


class Transformer4DDPM4(nn.Module):
    def __init__(
        self,
        in_channels=8,
        out_channels=5,
        resolution=32,
        dim_model=32,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        dim_feed_forward=2048,
        time_dim=256,
        encoding_layer=None,
        conditional=True,
        max_len=5000,
    ):
        super().__init__()

        # Layers
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.c_out = out_channels
        self.resolution = resolution
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.dim_feed_forward = dim_feed_forward
        self.time_dim = time_dim
        self.conditional = conditional
        self.max_len = max_len

        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout=dropout, max_len=max_len
        )
        # self.convert_to_state = nn.Linear(
        #     in_features=in_channels * resolution, out_features=out_channels * resolution
        # )

        self.post_freq_embedding_time = nn.Sequential(
            nn.Linear(self.time_dim, in_channels * resolution),
            nn.ReLU(),
        )

        self.output_first_token = nn.Parameter(torch.zeros(in_channels * resolution))

    def time_encoding(self, t, channels):
        inv_freq = (
            1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels)).cuda()
        )
        # repeat t tensor channels//2 times ona second dimension

        pos_enc_a = torch.sin(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, t):
        src = src.reshape((src.size(0), src.size(1), src.size(2) * src.size(3)))
        src = self.positional_encoder(src)
        # t = self.time_encoding(t, self.time_dim)
        # src += t

        return self.transformer.encoder(src, src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoder(tgt), memory, tgt_mask)

    def forward(
        self,
        x: torch.Tensor,
        t,
        y,
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):
        if self.conditional:
            t = self.time_encoding(t, self.time_dim).unsqueeze(1)
            tgt = x.reshape((x.size(0), x.size(1), x.size(2) * x.size(3)))
            tgt = torch.cat([t, tgt], dim=1)
            src = y.reshape((y.size(0), y.size(1), y.size(2) * y.size(3)))
        else:
            tgt = x.squeeze(2)
            src = self.time_encoding(t, self.time_dim).unsqueeze(1)
            tgt = self.positional_encoder(tgt)

        output_sequence = self.transformer(
            src,
            tgt,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        if self.conditional:
            out = output_sequence.reshape(
                (src.size(0), tgt.size(1), x.size(2), self.resolution)
            )[:, :, :5, :]
        else:
            out = output_sequence.reshape(
                (src.size(0), tgt.size(1), x.size(2), self.resolution)
            )

        return out


class Transformer4Diffusion(nn.Module):
    def __init__(
        self,
        in_channels=8,
        out_channels=5,
        resolution=32,
        dim_model=32,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        dim_feed_forward=2048,
        time_dim=256,
        conditional=True,
        max_len=5000,
        causal_decoder=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Layers
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.c_out = out_channels
        self.resolution = resolution
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.dim_feed_forward = dim_feed_forward
        self.time_dim = time_dim
        self.conditional = conditional
        self.max_len = max_len
        self.causal_decoder = causal_decoder

        self.input_emb = nn.Linear(
            self.input_channels * self.resolution, self.dim_model
        )
        self.last_norm = nn.LayerNorm(self.dim_model)
        self.head = nn.Linear(self.dim_model, self.c_out * self.resolution)

        self.time_emb = nn.Linear(self.time_dim, self.dim_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_feed_forward,
            dropout=dropout,
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=num_encoder_layers
        )

        if self.conditional:
            self.cond_emb = nn.Linear(
                self.input_channels * self.resolution, self.dim_model
            )
            self.input_emb = nn.Linear(
                self.output_channels * self.resolution, self.dim_model
            )
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dim_feedforward=dim_feed_forward,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=self.decoder_layer, num_layers=num_decoder_layers
            )

        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout=dropout, max_len=max_len
        )

        self.post_freq_embedding_time = nn.Sequential(
            nn.Linear(self.time_dim, in_channels * resolution),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        t,
        y,
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):
        t = time_encoding(t, self.time_dim).unsqueeze(1)
        t = self.time_emb(t)

        if self.conditional:
            src = y.reshape(y.size(0), y.size(1), y.size(2) * y.size(3))
            src = self.cond_emb(src)
            src = torch.cat([t, src], dim=1)
            if src_padding_mask is not None:
                src_padding_mask = torch.cat(
                    [
                        torch.zeros(src_padding_mask.size(0), 1).to(
                            src_padding_mask.device
                        ),
                        src_padding_mask,
                    ],
                    dim=1,
                )

            tgt = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3))
            tgt = self.input_emb(tgt)

            src = self.positional_encoder(src)
            tgt = self.positional_encoder(tgt)

            memory = self.encoder(
                src,
                src_key_padding_mask=src_padding_mask,
            )

            output_sequence = self.decoder(
                tgt,
                memory,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask,
                # is_causal=self.causal_decoder,
            )

            output_sequence = self.head(self.last_norm(output_sequence))

            out = output_sequence.reshape(
                (x.size(0), x.size(1), x.size(2), self.resolution)
            )

            return out

            # src = y.reshape(y.size(0), y.size(1), y.size(2) * y.size(3))
            # src = self.cond_emb(src)
            # # src = torch.cat([t, src], dim=1)

            # tgt = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3))
            # tgt = self.input_emb(tgt)
            # tgt = torch.cat([t, tgt], dim=1)
            # if tgt_padding_mask is not None:
            #     tgt_padding_mask = torch.cat(
            #         [
            #             torch.zeros(tgt_padding_mask.size(0), 1).to(
            #                 tgt_padding_mask.device
            #             ),
            #             tgt_padding_mask,
            #         ],
            #         dim=1,
            #     )

            # src = self.positional_encoder(src)
            # tgt = self.positional_encoder(tgt)

            # memory = self.encoder(
            #     src,
            #     src_key_padding_mask=src_padding_mask,
            # )

            # output_sequence = self.decoder(
            #     tgt,
            #     memory,
            #     tgt_key_padding_mask=tgt_padding_mask,
            #     memory_key_padding_mask=src_padding_mask,
            #     # is_causal=self.causal_decoder,
            # )

            # output_sequence = self.head(self.last_norm(output_sequence))

            # out = output_sequence.reshape(
            #     (x.size(0), x.size(1) + 1, x.size(2), self.resolution)
            # )

            # return out[:, 1:, :, :]
        else:
            src = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3))
            src = self.input_emb(src)
            src = torch.cat([t, src], dim=1)

            src = self.positional_encoder(src)

            output_sequence = self.encoder(
                src,
                src_key_padding_mask=src_padding_mask,
            )[:, 1:, :]

            output_sequence = self.head(self.last_norm(output_sequence))

            out = output_sequence.reshape(
                (x.size(0), x.size(1), x.size(2), self.resolution)
            )

            return out


class Transformer4DDPM3(nn.Module):
    def __init__(
        self,
        in_channels=8,
        out_channels=5,
        resolution=32,
        dim_model=32,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        dim_feed_forward=2048,
        time_dim=256,
        encoding_layer=None,
        conditional=True,
    ):
        super().__init__()

        # Layers
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.c_out = out_channels
        self.resolution = resolution
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.dim_feed_forward = dim_feed_forward
        self.time_dim = time_dim
        self.conditional = conditional

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
        # self.convert_to_state = nn.Linear(
        #     in_features=in_channels * resolution, out_features=out_channels * resolution
        # )

        self.post_freq_embedding_time = nn.Sequential(
            nn.Linear(self.time_dim, in_channels * resolution),
            nn.ReLU(),
        )

        self.output_first_token = nn.Parameter(torch.zeros(in_channels * resolution))

    def time_encoding(self, t, channels):
        inv_freq = (
            1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels)).cuda()
        )
        # repeat t tensor channels//2 times ona second dimension

        pos_enc_a = torch.sin(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, t):
        src = src.reshape((src.size(0), src.size(1), src.size(2) * src.size(3)))
        src = self.positional_encoder(src)
        # t = self.time_encoding(t, self.time_dim)
        # src += t

        return self.transformer.encoder(src, src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoder(tgt), memory, tgt_mask)

    def forward(
        self,
        x: torch.Tensor,
        t,
        y,
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):
        tgt = x

        src = self.time_encoding(t, self.time_dim).unsqueeze(1)
        # src = self.post_freq_embedding_time(src)

        # tgt_start = torch.cat([t.unsqueeze(1)] * tgt.size(1), dim=1)

        src = self.positional_encoder(src)
        # tgt = torch.cat([src, tgt], dim=1)
        tgt = self.positional_encoder(tgt)

        output_sequence = self.transformer(
            src,
            tgt,
        )

        if self.conditional:
            # out = self.convert_to_state(output_sequence).reshape(
            #     (src.size(0), tgt.size(1), self.output_channels, self.resolution)
            # )
            out = output_sequence.reshape(
                (src.size(0), tgt.size(1), initial_shape[2], self.resolution)
            )[:, :, :5, :]
        else:
            out = output_sequence.reshape((src.size(0), tgt.size(1), self.resolution))

        return out


class Transformer4DDPM(nn.Module):
    def __init__(
        self,
        in_channels=8,
        out_channels=5,
        resolution=32,
        dim_model=32,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        dim_feed_forward=2048,
        time_dim=256,
        encoding_layer=None,
        conditional=True,
    ):
        super().__init__()

        # Layers
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.resolution = resolution
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.dim_feed_forward = dim_feed_forward
        self.time_dim = time_dim
        self.conditional = conditional

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
        # self.convert_to_state = nn.Linear(
        #     in_features=in_channels * resolution, out_features=out_channels * resolution
        # )

        self.post_freq_embedding_time = nn.Sequential(
            nn.Linear(self.time_dim, in_channels * resolution),
            nn.ReLU(),
        )

        self.output_first_token = nn.Parameter(torch.zeros(in_channels * resolution))

    def time_encoding(self, t, channels):
        inv_freq = (
            1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels)).cuda()
        )
        # repeat t tensor channels//2 times ona second dimension

        pos_enc_a = torch.sin(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor, t):
        src = src.reshape((src.size(0), src.size(1), src.size(2) * src.size(3)))
        src = self.positional_encoder(src)
        # t = self.time_encoding(t, self.time_dim)
        # src += t

        return self.transformer.encoder(src, src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoder(tgt), memory, tgt_mask)

    def forward(
        self,
        past_surfaces: torch.Tensor,
        observations: torch.Tensor,
        next_surfaces: torch.Tensor,
        t,
        src_padding_mask=None,
        tgt_padding_mask=None,
        next_noises: torch.Tensor = None,
    ):
        if self.conditional:
            src = torch.cat([past_surfaces, observations], dim=2)
            tgt = torch.cat([next_surfaces, observations], dim=2)
        else:
            src = past_surfaces
            tgt = next_noises
        initial_shape = src.shape

        src = src.reshape((src.size(0), src.size(1), src.size(2) * src.size(3)))
        if next_noises is not None:
            tgt = tgt.reshape((tgt.size(0), tgt.size(1), tgt.size(2) * tgt.size(3)))

        # tgt_start = self.output_first_token.repeat(src.size(0), 1).unsqueeze(1)
        # tgt = torch.cat([tgt_start, tgt[:, :-1, :]], dim=1)

        # src += t.unsqueeze(1)
        # tgt += t.unsqueeze(1)

        t = self.time_encoding(t, self.time_dim)
        # t = self.post_freq_embedding_time(t)

        # tgt_start = torch.cat([t.unsqueeze(1)] * tgt.size(1), dim=1)
        if next_noises is not None:
            tgt = torch.cat([t.unsqueeze(1), tgt], dim=1)
        else:
            tgt = t.unsqueeze(1)

        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        first_true = torch.zeros(src.size(0), 1).to(src.device)
        if next_noises is not None:
            tgt_padding_mask = torch.cat([first_true, tgt_padding_mask], dim=1)
        else:
            tgt_padding_mask = first_true

        src_mask, tgt_mask = create_mask(src, tgt, device=src.device)

        output_sequence = self.transformer(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        if self.conditional:
            # out = self.convert_to_state(output_sequence).reshape(
            #     (src.size(0), tgt.size(1), self.output_channels, self.resolution)
            # )
            out = output_sequence.reshape(
                (src.size(0), tgt.size(1), initial_shape[2], self.resolution)
            )[:, :, :5, :]
        else:
            out = output_sequence.reshape(
                (src.size(0), tgt.size(1), self.output_channels, self.resolution)
            )

        return out

    def generate_out_sequence(self, past_surfaces, observations, t):
        src = past_surfaces

        len_out_seq = src.size(1)

        out_sequence = None

        out_sequence_converted = torch.Tensor().cuda()

        for i in range(len_out_seq):
            src_padding_mask = torch.zeros(src.size(0), src.size(1)).cuda()
            tgt_padding_mask = torch.zeros(src.size(0), i).cuda()
            out = self.forward(
                src,
                observations,
                None,
                t,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask if i > 0 else None,
                next_noises=out_sequence,
            )
            out_sequence = (
                torch.cat([out_sequence, out[:, -1:, :]], dim=1)
                if out_sequence is not None
                else out[:, -1:, :]
            )
            if self.conditional:
                out_sequence_converted = torch.cat(
                    [out_sequence_converted, out[:, -1:, :]],
                    dim=1,
                )
            else:
                out_sequence_converted = torch.cat(
                    [out_sequence_converted, out[:, -1:, :]], dim=1
                )

        return out_sequence_converted.reshape(
            (
                out_sequence.size(0),
                out_sequence.size(1),
                src.size(2),
                self.resolution,
            )
        )[:, :, :5, :]

    def generate_out_sequence2(self, past_surfaces, observations, t):
        if self.conditional:
            src = torch.cat([past_surfaces, observations], dim=2)
        else:
            src = past_surfaces

        len_out_seq = src.size(1)

        src_mask = (
            torch.zeros((len_out_seq, len_out_seq))
            .to(past_surfaces.device)
            .type(torch.bool)
        )

        memory = self.encode(src, src_mask, t)
        # out_sequence = (
        #     torch.cat([self.output_first_token, observations[0]], dim=1)
        #     .repeat(src.size(0), 1)
        #     .unsqueeze(1)
        # )
        # out_sequence = self.output_first_token.unsqueeze(0).unsqueeze(0)

        t = self.time_encoding(t, self.time_dim)
        t = self.post_freq_embedding_time(t)

        out_sequence = t.unsqueeze(1)

        if self.conditional:
            # out_sequence_converted = self.convert_to_state(out_sequence)
            out_sequence_converted = out_sequence
        else:
            out_sequence_converted = out_sequence

        for i in range(len_out_seq):
            tgt_mask = generate_square_subsequent_mask(out_sequence.size(1))
            out = self.decode(out_sequence, memory, tgt_mask)
            out_sequence = torch.cat([out_sequence, out[:, -1:, :]], dim=1)
            if self.conditional:
                out_sequence_converted = torch.cat(
                    [out_sequence_converted, out[:, -1:, :]],
                    dim=1,
                )
            else:
                out_sequence_converted = torch.cat(
                    [out_sequence_converted, out[:, -1:, :]], dim=1
                )

        return out_sequence_converted.reshape(
            (
                out_sequence.size(0),
                out_sequence.size(1),
                src.size(2),
                self.resolution,
            )
        )[:, 1:, :5, :]


class TransformerAlone(nn.Module):
    def __init__(
        self,
        in_channels=5,
        out_channels=5,
        resolution=32,
        dim_model=32,
        num_heads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        dim_feed_forward=2048,
        time_dim=256,
        encoding_layer=None,
        conditional=True,
    ):
        super().__init__()

        # Layers
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.resolution = resolution
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.dim_feed_forward = dim_feed_forward
        self.time_dim = time_dim
        self.conditional = conditional

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
        self.first_convert = nn.Linear(
            in_features=out_channels * resolution, out_features=in_channels * resolution
        )
        self.convert_to_state = nn.Linear(
            in_features=in_channels * resolution, out_features=out_channels * resolution
        )

        self.output_first_token = nn.Parameter(torch.zeros(in_channels * resolution))

    def time_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float() / channels)
        ).to(self.device)
        # repeat t tensor channels//2 times ona second dimension

        pos_enc_a = torch.sin(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.unsqueeze(1).repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        src = src.reshape((src.size(0), src.size(1), src.size(2) * src.size(3)))
        src = self.positional_encoder(src)

        return self.transformer.encoder(src, src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoder(tgt), memory, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):
        initial_shape = src.shape
        src = src.reshape((src.size(0), src.size(1), src.size(2) * src.size(3)))
        tgt = tgt.reshape((tgt.size(0), tgt.size(1), tgt.size(2) * tgt.size(3)))
        src = self.positional_encoder(src)
        tgt = self.first_convert(tgt)
        tgt = self.positional_encoder(tgt)

        tgt_start = self.output_first_token.repeat(src.size(0), 1).unsqueeze(1)
        tgt = torch.cat([tgt_start, tgt], dim=1)

        first_true = torch.zeros(src.size(0), 1).to(src.device)
        tgt_padding_mask = torch.cat([first_true, tgt_padding_mask], dim=1)

        src_mask, tgt_mask = create_mask(src, tgt, device=src.device)

        output_sequence = self.transformer(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        if self.conditional:
            out = self.convert_to_state(output_sequence).reshape(
                (src.size(0), tgt.size(1), self.output_channels, self.resolution)
            )
        else:
            out = output_sequence.reshape(
                (src.size(0), tgt.size(1), self.output_channels, self.resolution)
            )

        return out

    def generate_out_sequence(self, past_surfaces, observations):
        if self.conditional:
            src = observations.cuda()
        else:
            src = past_surfaces

        len_out_seq = src.size(1)

        src_mask = torch.zeros((len_out_seq, len_out_seq)).cuda().type(torch.bool)

        memory = self.encode(src, src_mask)
        # out_sequence = (
        #     torch.cat([self.output_first_token, observations[0]], dim=1)
        #     .repeat(src.size(0), 1)
        #     .unsqueeze(1)
        # )
        out_sequence = self.output_first_token.unsqueeze(0).unsqueeze(0)

        if self.conditional:
            out_sequence_converted = self.convert_to_state(out_sequence)
            # out_sequence_converted = out_sequence
        else:
            out_sequence_converted = out_sequence

        for i in range(len_out_seq):
            tgt_mask = generate_square_subsequent_mask(out_sequence.size(1))
            out = self.decode(out_sequence, memory, tgt_mask)
            out_sequence = torch.cat([out_sequence, out[:, -1:, :]], dim=1)
            if self.conditional:
                out_sequence_converted = torch.cat(
                    [out_sequence_converted, self.convert_to_state(out[:, -1:, :])],
                    dim=1,
                )
            else:
                out_sequence_converted = torch.cat(
                    [out_sequence_converted, out[:, -1:, :]], dim=1
                )

        return out_sequence_converted.reshape(
            (
                out_sequence.size(0),
                out_sequence.size(1),
                5,
                self.resolution,
            )
        )[:, 1:, :, :]


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout, max_len, learnable=False):
        super().__init__()
        den = torch.exp(-torch.arange(0, dim_model, 2) * math.log(10000) / dim_model)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, dim_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.register_buffer("pos_embedding", pos_embedding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(
            token_embedding
            + self.pos_embedding[: token_embedding.size(1), :].repeat(
                (token_embedding.size(0), 1, 1)
            )
        )
