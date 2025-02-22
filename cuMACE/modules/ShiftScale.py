import torch
from torch import nn


@compile_mode("script")
class ShiftScaleBlock(torch.nn.Module):
   """
    ShiftScaleBlock for atom-wise quantity 

    Attributes:
        Zs: 

    Methods:
      forward
   """
   
   def __init__(self, shift: torch.tensor, scale: torch.tensor  ):
        super().__init__()
        self.register_buffer(
            "scale",
            torch.tensor(scale, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "shift",
            torch.tensor(shift, dtype=torch.get_default_dtype()),
        )

   def forward(self, x: torch.Tensor, atom_type: torch.Tensor) -> torch.Tensor:
        #
        return (
            torch.atleast_1d(self.scale)[atom_type] * x + torch.atleast_1d(self.shift)[atom_type]
        )