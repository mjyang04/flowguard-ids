from __future__ import annotations

import torch.nn as nn


class BaseNIDSModel(nn.Module):
    def get_metadata(self) -> dict:
        return {}
