import torch
from torch import nn


class PPnet(nn.Module):
    def __init__(self, input_size=6, output_size=6, seq=20, hidden_size=8, num_layer=1, batch_first=True, nhead=6, model_type="lstm"):
        super().__init__()

        self.hidden_size = hidden_size
        self.seq = seq
        self.type = model_type

        if self.type == "gru":
            self.model = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=batch_first)
        elif self.type == "sgru":
            self.model = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer+1, batch_first=batch_first)
        elif self.type == "lstm":
            self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=batch_first)
        elif self.type == "slstm":
            self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer+1, batch_first=batch_first)
        elif self.type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
            self.model = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        elif self.type == "stransformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
            self.model = nn.TransformerEncoder(encoder_layer, num_layers=num_layer+1)

        self.fn_mean = nn.Linear(seq * hidden_size, output_size)
        self.fn_variance = nn.Linear(seq * hidden_size, output_size)

    def forward(self, x):
        if "lstm" in self.type or "gru" in self.type:
            t_feature, _ = self.model(x)
        else:
            # TODO: Sahiti
            t_feature = self.model(x)

        t_feature = t_feature.reshape(-1, self.hidden_size * self.seq)

        mean = self.fn_mean(t_feature)             # [BS, 6]
        variance = self.fn_variance(t_feature)     # [BS, 6]

        return mean, variance
