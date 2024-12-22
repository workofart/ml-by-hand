from autograd.nn import Module, Conv2d
from autograd.functional import relu


class ResidualBlock(Module):
    """
    Residual Block as described in Deep Residual Learning for Image Recognition
    The residual block as a whole implements both F(x) and x (identity mapping)
    The function that were trying to learn is H(x) = F(x) + x
    F(x) is the convolutional layer with ReLU activation
    x is the identity mapping

    Paper: https://arxiv.org/abs/1512.03385
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding_mode="same"
        )
        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding_mode="same",
        )

        # Add projection shortcut when dimensions change
        self.shortcut = Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, padding_mode="same"
        )

    def forward(self, x):
        identity = self.shortcut(x)  # Match channels

        out = self.conv1(x)
        out = relu(out)
        out = self.conv2(out)
        return relu(out) + identity
