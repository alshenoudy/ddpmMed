"""
Based on nnUNet metrics.py and modified a bit ofr our use case
https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/evaluation/metrics.py
"""

import os
import numpy as np
from medpy import metric

