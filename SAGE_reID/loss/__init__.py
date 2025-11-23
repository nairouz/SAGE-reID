from __future__ import absolute_import

from .triplet import SoftTripletLoss_vallia, SoftTripletLoss, TripletLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy
from .invariance import InvNet
__all__ = [
    'SoftTripletLoss_vallia',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy',
    'InvNet',
    'TripletLoss'
]
