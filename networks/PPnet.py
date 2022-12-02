import torch
from torch import nn


class PPnet(nn.Module):
    def __init__(self, input_size=6, output_size=6, seq=20, hidden_size=8, num_layer=1, batch_first=True, model_type="lstm"):
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
        else:
            # TODO: Sahiti
            self.model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1)

        self.fn_mean = nn.Linear(seq * hidden_size, output_size)
        self.fn_variance = nn.Linear(seq * hidden_size, output_size)

    def forward(self, x):
        t_feature, (_, _) = self.model(x)

        t_feature = t_feature.reshape(-1, self.hidden_size * self.seq)

        mean = self.fn_mean(t_feature)             # [BS, 6]
        variance = self.fn_variance(t_feature)     # [BS, 6]

        return mean, variance
