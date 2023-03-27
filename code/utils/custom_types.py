import torch
from collections import OrderedDict
from typing import NewType, Any

ParamDict = NewType("ParamDict", OrderedDict[str, torch.Tensor])
TrajDataDict = NewType("TrajDataDict", dict[str, list[torch.Tensor, ...]])
PolicyContext = NewType("PolicyContext", dict[str, Any])
NNLayerSizes = NewType("NNLayerSizes", tuple[int, ...])

