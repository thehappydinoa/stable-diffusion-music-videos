import torch
from diffusers import ModelMixin


class NoCheck(ModelMixin):
    """Can be used in place of safety checker. Use responsibly and at your own risk."""

    def __init__(self):
        super().__init__()
        self.register_parameter(name="asdf", param=torch.nn.Parameter(torch.randn(3)))

    def forward(self, images=None, **kwargs):
        return images, [False]
