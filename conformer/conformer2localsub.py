import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from encoder import ConformerEncoder
from modules import Linear


class ConformerWithTwoLocallyConnectedLayers(nn.Module):
    """
    Conformer with Two Locally Connected Layers in the Input Stage.

    Args:
        num_classes (int): Number of classification classes.
        input_dim (int, optional): Dimension of input vector.
        encoder_dim (int, optional): Dimension of conformer encoder.
        num_encoder_layers (int, optional): Number of conformer blocks.
        num_attention_heads (int, optional): Number of attention heads.
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module.
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module.
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout.
        attention_dropout_p (float, optional): Probability of attention module dropout.
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout.
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel.
        half_step_residual (bool): Flag indication whether to use half step residual or not.

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector.
        - **input_lengths** (batch): list of sequence input lengths.

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produced by conformer.
        - **output_lengths** (batch): list of sequence output lengths.
    """
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 80,
        encoder_dim: int = 256,
        num_encoder_layers: int = 4,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ) -> None:
        super(ConformerWithTwoLocallyConnectedLayers, self).__init__()

        # First locally connected layer
        self.locally_connected1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=encoder_dim // 2,  # Reduced to fit groups
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,  # Disable grouping
        )
        self.relu1 = nn.ReLU()

        # Second locally connected layer
        self.locally_connected2 = nn.Conv1d(
            in_channels=encoder_dim // 2,
            out_channels=encoder_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,  # Disable grouping
        )
        self.relu2 = nn.ReLU()

        self.encoder = ConformerEncoder(
            input_dim=encoder_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        )
        self.fc = Linear(encoder_dim, num_classes, bias=False)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return self.encoder.count_parameters()

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate inputs through the locally connected layers and encoder.

        Args:
            inputs (torch.FloatTensor): Input sequence passed to the model.
            input_lengths (torch.LongTensor): Length of input tensor.

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        # Locally connected layers
        inputs = inputs.permute(0, 2, 1)  # [B, T, D] -> [B, D, T]
        inputs = self.locally_connected1(inputs)
        inputs = self.relu1(inputs)
        inputs = self.locally_connected2(inputs)
        inputs = self.relu2(inputs)

        # Pass through encoder
        inputs = inputs.permute(0, 2, 1)  # [B, D, T] -> [B, T, D]
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs = self.fc(encoder_outputs)
        outputs = nn.functional.log_softmax(outputs, dim=-1)
        return outputs, encoder_output_lengths