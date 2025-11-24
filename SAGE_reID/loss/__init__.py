from __future__ import absolute_import

from .triplet import SoftTripletLoss_vallia, SoftTripletLoss, TripletLoss, TripletLossCD, SoftTripletLossCD
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy, CrossEntropyLabelSmoothCD
from .invariance import InvNet
__all__ = [
    'SoftTripletLoss_vallia',
    'CrossEntropyLabelSmooth',
    'CrossEntropyLabelSmoothCD',
    'SoftTripletLoss',
    'SoftEntropy',
    'InvNet',
    'TripletLoss',
    'TripletLossCD',
    'SoftTripletLossCD'
]
