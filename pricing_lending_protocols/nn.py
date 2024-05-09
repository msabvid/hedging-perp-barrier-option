from typing import List

import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(
        self, sizes, activation=nn.ReLU, output_activation=nn.Identity, batch_norm=False
    ):
        super().__init__()

        layers = (
            [
                nn.BatchNorm1d(sizes[0]),
            ]
            if batch_norm
            else []
        )
        for j in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[j], sizes[j + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(sizes[j + 1], affine=True))
            if j < (len(sizes) - 2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RNN(nn.Module):
    def __init__(
        self,
        rnn_in: int,
        rnn_hidden: int,
        ffn_sizes: List[int],
        activation=nn.ReLU,
        output_activation=nn.Identity,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=rnn_in, hidden_size=rnn_hidden, num_layers=1, batch_first=True
        )
        layers = []
        for j in range(len(ffn_sizes) - 1):
            layers.append(nn.Linear(ffn_sizes[j], ffn_sizes[j + 1]))
            if j < (len(ffn_sizes) - 2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.ffn = nn.Sequential(*layers)

    def forward(self, *x):
        """Forward method

        Parameters
        ----------
        x: torch.Tensor
            Sequential input. Tensor of size (N,L,d) where N is batch size, L is lenght of the sequence, and d is dimension of the path
        Returns
        -------
        torch.Tensor
            Sequential output. Tensor of size (N, L, d_out) containing the output from the last layer of the RNN for each timestep
        """
        output_RNN, _ = self.rnn(torch.cat(x, -1))
        output = self.ffn(output_RNN)
        return output


class Linear(nn.Module):
    def __init__(self, ffn_sizes: List[int]):

        super().__init__()
        self.net = nn.Linear(*ffn_sizes)

    def forward(self, x):
        return self.net(x)
