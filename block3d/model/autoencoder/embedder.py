import math

import torch
import torch.nn as nn


class PhaseModulatedFourierEmbedder(torch.nn.Module):
    def __init__(
        self,
        num_freqs: int,
        input_dim: int = 3,
    ):
        """
        Initializes the PhaseModulatedFourierEmbedder class.
        Args:
            num_freqs (int): The number of frequencies to be used.
            input_dim (int, optional): The dimension of the input. Defaults to 3.
        Attributes:
            weight (torch.nn.Parameter): The weight parameter initialized with random values.
            carrier (torch.Tensor): The carrier frequencies calculated based on the Nyquist-Shannon sampling theorem.
            out_dim (int): The output dimension calculated based on the input dimension and number of frequencies.
        """

        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(input_dim, num_freqs) * math.sqrt(0.5 * num_freqs)
        )

        # Highest useful frequency under the sampling layout used by the tokenizer.
        carrier = (num_freqs / 8) ** torch.linspace(1, 0, num_freqs)
        carrier = (carrier + torch.linspace(0, 1, num_freqs)) * 2 * torch.pi
        self.register_buffer("carrier", carrier, persistent=False)

        self.out_dim = input_dim * (num_freqs * 2 + 1)

    def forward(self, x):
        """
        Perform the forward pass of the embedder model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, ..., output_dim) where
                          output_dim = input_dim + 2 * input_dim.
        """

        m = x.float().unsqueeze(-1)
        fm = (m * self.weight).view(*x.shape[:-1], -1)
        pm = (m * 0.5 * torch.pi + self.carrier).view(*x.shape[:-1], -1)
        embedding = torch.cat([x, fm.cos() + pm.cos(), fm.sin() + pm.sin()], dim=-1)

        return embedding
