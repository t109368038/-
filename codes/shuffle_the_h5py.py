#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 22:01:49 2017  

@author: wu
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io
#import random
import h5py
from random import shuffle


dest = '/home/lab21010/h5py_files/'
use_file = {"train": ["0_8_4", "0_3_1", "0_2_0", "0_13_5", "0_2_19", "0_3_2", "0_12_1", "0_9_7", "0_3_22", "0_2_2", "0_2_3", "0_5_8", "0_8_21", "0_9_4", "0_2_17", "0_13_13", "0_2_22", "0_9_15", "0_9_17", "0_11_14", "0_10_14", "0_2_16", "0_6_7", "0_2_10", "0_12_19", "0_8_24", "0_8_3", "0_11_4", "0_2_11", "0_3_3", "0_12_14", "0_5_7", "0_11_3", "0_10_1", "0_9_14", "0_11_23", "0_12_13", "0_8_19", "0_8_16", "0_10_16", "0_9_2", "0_8_13", "0_3_9", "0_9_12", "0_13_12", "0_5_15", "0_6_15", "0_6_0", "0_5_0", "0_10_7", "0_9_3", "0_3_11", "0_3_6", "0_13_18", "0_6_19", "0_9_5", "0_10_11", "0_12_7", "0_12_16", "0_11_16", "0_8_14", "0_11_20", "0_10_0", "0_10_13", "0_6_17", "0_6_11", "0_11_17", "0_10_2", "0_11_2", "0_12_18", "0_12_4", "0_11_9", "0_2_14", "0_6_6", "0_5_9", "0_10_8", "0_8_23", "0_11_0", "0_3_20", "0_11_7", "0_11_13", "0_2_7", "0_11_5", "0_8_12", "0_10_21", "0_2_9", "0_12_5", "0_3_23", "0_3_19", "0_8_7", "0_10_10", "0_5_2", "0_13_22", "0_8_5", "0_11_22", "0_11_24", "0_9_24", "0_3_12", "0_2_6", "0_8_20", "0_5_18", "0_5_3", "0_3_8", "0_9_9", "0_11_19", "0_13_23", "0_6_5", "0_11_18", "0_3_13", "0_3_16", "0_9_1", "0_3_4", "0_5_6", "0_6_22", "0_13_20", "0_9_10", "0_3_17", "0_6_10", "0_2_24", "0_10_3", "0_5_5", "0_6_18", "0_6_4", "0_2_1", "0_3_14", "0_10_5", "1_8_3", "1_2_24", "1_2_16", "1_9_12", "1_6_14", "1_13_24", "1_3_6", "1_3_1", "1_9_7", "1_3_24", "1_5_14", "1_9_6", "1_13_6", "1_13_3", "1_2_6", "1_6_10", "1_10_10", "1_2_12", "1_3_4", "1_5_12", "1_12_7", "1_12_24", "1_5_21", "1_12_12", "1_11_9", "1_11_10", "1_13_10", "1_11_19", "1_6_4", "1_5_1", "1_3_16", "1_5_6", "1_6_15", "1_5_18", "1_2_13", "1_9_21", "1_13_19", "1_3_0", "1_13_1", "1_10_12", "1_12_5", "1_9_13", "1_11_22", "1_3_10", "1_3_14", "1_8_11", "1_5_15", "1_6_2", "1_5_9", "1_9_24", "1_8_10", "1_5_10", "1_6_0", "1_10_8", "1_10_2", "1_8_19", "1_13_5", "1_12_11", "1_12_22", "1_9_17", "1_2_18", "1_11_4", "1_2_0", "1_5_22", "1_3_15", "1_10_20", "1_2_5", "1_2_8", "1_13_2", "1_10_5", "1_11_3", "1_10_24", "1_10_21", "1_6_11", "1_11_14", "1_9_1", "1_8_22", "1_13_21", "1_10_14", "1_9_19", "1_6_16", "1_2_10", "1_3_21", "1_8_14", "1_10_1", "1_12_14", "1_2_17", "1_5_19", "1_13_9", "1_8_12", "1_2_11", "1_13_11", "1_3_17", "1_11_17", "1_12_2", "1_12_15", "1_3_9", "1_10_6", "1_11_15", "1_2_23", "1_2_14", "1_8_16", "1_6_3", "1_3_5", "1_11_24", "1_13_14", "1_13_20", "1_3_2", "1_10_18", "1_9_16", "1_9_23", "1_8_2", "1_12_18", "1_3_8", "1_13_7", "1_2_4", "1_2_22", "1_5_8", "1_6_24", "1_2_9", "1_2_2", "1_11_12", "1_11_21", "1_11_8", "1_8_5", "1_11_18", "2_6_6", "2_9_20", "2_2_7", "2_3_16", "2_8_24", "2_8_9", "2_8_21", "2_11_17", "2_8_23", "2_9_8", "2_3_8", "2_2_11", "2_8_18", "2_13_11", "2_9_5", "2_11_14", "2_13_9", "2_12_14", "2_8_16", "2_10_21", "2_2_13", "2_13_18", "2_2_23", "2_12_15", "2_8_12", "2_11_0", "2_5_5", "2_5_19", "2_8_13", "2_5_12", "2_13_21", "2_3_13", "2_9_22", "2_2_9", "2_10_20", "2_6_2", "2_5_3", "2_10_17", "2_11_24", "2_10_12", "2_11_16", "2_11_5", "2_3_0", "2_12_19", "2_3_2", "2_10_0", "2_10_18", "2_13_2", "2_12_17", "2_12_11", "2_13_10", "2_5_17", "2_13_7", "2_11_8", "2_5_6", "2_13_20", "2_8_7", "2_6_7", "2_12_22", "2_6_16", "2_5_21", "2_3_5", "2_5_11", "2_13_17", "2_3_1", "2_6_12", "2_10_24", "2_3_3", "2_8_2", "2_10_23", "2_3_7", "2_5_1", "2_8_11", "2_5_15", "2_6_23", "2_2_5", "2_12_23", "2_5_23", "2_10_22", "2_2_3", "2_13_1", "2_9_7", "2_12_10", "2_12_9", "2_11_6", "2_2_18", "2_12_2", "2_10_10", "2_2_2", "2_11_2", "2_3_20", "2_13_6", "2_12_5", "2_11_18", "2_6_14", "2_5_20", "2_5_8", "2_8_3", "2_10_5", "2_8_4", "2_9_17", "2_2_14", "2_10_19", "2_2_22", "2_2_12", "2_11_21", "2_12_7", "2_6_5", "2_6_13", "2_12_24", "2_5_14", "2_6_8", "2_8_0", "2_8_1", "2_10_7", "2_12_3", "2_3_22", "2_9_24", "2_3_23", "2_12_20", "2_10_11", "2_3_19", "2_5_13", "2_11_9", "2_9_23", "2_2_20", "3_5_21", "3_8_14", "3_9_1", "3_10_21", "3_9_14", "3_12_1", "3_8_24", "3_3_4", "3_12_3", "3_9_24", "3_13_4", "3_2_2", "3_12_14", "3_8_4", "3_6_22", "3_10_16", "3_10_2", "3_5_4", "3_3_13", "3_2_21", "3_11_0", "3_6_16", "3_3_2", "3_8_12", "3_12_18", "3_5_2", "3_6_12", "3_11_13", "3_10_6", "3_6_8", "3_6_21", "3_10_19", "3_11_2", "3_3_5", "3_12_19", "3_13_18", "3_12_12", "3_13_22", "3_3_22", "3_10_8", "3_6_17", "3_11_10", "3_2_10", "3_5_14", "3_6_18", "3_2_9", "3_5_22", "3_9_3", "3_3_8", "3_12_20", "3_2_12", "3_10_1", "3_13_9", "3_13_12", "3_11_20", "3_12_10", "3_6_24", "3_13_2", "3_5_6", "3_12_21", "3_10_17", "3_10_18", "3_9_23", "3_3_12", "3_12_11", "3_8_17", "3_11_6", "3_8_8", "3_11_7", "3_3_19", "3_10_22", "3_10_14", "3_10_20", "3_13_15", "3_5_8", "3_3_10", "3_8_20", "3_11_9", "3_9_13", "3_6_11", "3_6_19", "3_11_18", "3_13_5", "3_8_9", "3_13_21", "3_9_21", "3_2_3", "3_9_7", "3_2_11", "3_3_20", "3_5_15", "3_12_7", "3_5_5", "3_3_17", "3_10_13", "3_11_19", "3_9_4", "3_3_14", "3_10_7", "3_9_11", "3_10_11", "3_10_10", "3_13_17", "3_5_19", "3_2_23", "3_6_5", "3_3_11", "3_13_14", "3_8_10", "3_5_11", "3_9_22", "3_13_13", "3_3_16", "3_11_5", "3_3_18", "3_6_23", "3_5_17", "3_13_1", "3_9_20", "3_5_24", "3_12_22", "3_6_6", "3_12_16", "3_12_15", "3_10_3", "3_5_16", "4_3_14", "4_3_7", "4_3_20", "4_3_5", "4_5_8", "4_2_15", "4_6_10", "4_13_9", "4_13_23", "4_10_16", "4_5_16", "4_10_3", "4_3_1", "4_10_24", "4_2_13", "4_5_11", "4_6_23", "4_9_1", "4_10_23", "4_6_20", "4_2_22", "4_11_18", "4_5_15", "4_9_22", "4_10_20", "4_11_7", "4_12_4", "4_3_0", "4_9_15", "4_9_17", "4_9_5", "4_9_21", "4_9_4", "4_13_24", "4_9_24", "4_10_11", "4_3_9", "4_6_2", "4_9_6", "4_3_3", "4_8_15", "4_3_18", "4_6_14", "4_13_11", "4_8_19", "4_5_4", "4_3_16", "4_6_6", "4_5_7", "4_8_14", "4_11_12", "4_12_14", "4_11_22", "4_11_17", "4_8_24", "4_6_1", "4_13_20", "4_12_21", "4_8_7", "4_3_22", "4_12_17", "4_10_9", "4_11_19", "4_3_17", "4_2_16", "4_10_19", "4_13_21", "4_9_20", "4_8_10", "4_5_2", "4_9_3", "4_9_16", "4_2_7", "4_13_3", "4_10_0", "4_8_22", "4_11_23", "4_5_0", "4_6_0", "4_11_6", "4_9_10", "4_2_6", "4_9_18", "4_8_23", "4_5_19", "4_2_14", "4_12_19", "4_11_4", "4_8_13", "4_8_2", "4_5_18", "4_13_15", "4_13_19", "4_8_5", "4_11_3", "4_3_19", "4_2_23", "4_2_8", "4_3_12", "4_6_18", "4_12_9", "4_2_4", "4_11_0", "4_12_5", "4_12_23", "4_6_15", "4_11_5", "4_3_13", "4_5_24", "4_12_18", "4_13_7", "4_5_12", "4_8_9", "4_6_8", "4_2_3", "4_10_17", "4_5_1", "4_6_24", "4_9_14", "4_3_8", "4_9_19", "4_10_18", "4_5_14", "4_10_1", "4_9_23", "4_12_13", "5_10_0", "5_3_24", "5_12_1", "5_12_4", "5_2_10", "5_8_22", "5_2_21", "5_9_7", "5_6_21", "5_12_14", "5_13_3", "5_12_5", "5_11_8", "5_10_11", "5_9_20", "5_11_1", "5_8_1", "5_12_9", "5_12_22", "5_5_17", "5_6_15", "5_9_23", "5_8_16", "5_6_19", "5_2_11", "5_13_4", "5_11_5", "5_3_20", "5_13_0", "5_8_13", "5_6_5", "5_2_19", "5_9_3", "5_5_10", "5_5_0", "5_9_16", "5_6_0", "5_2_15", "5_5_12", "5_2_4", "5_2_7", "5_3_2", "5_3_11", "5_11_11", "5_3_8", "5_11_13", "5_13_21", "5_5_15", "5_12_24", "5_3_7", "5_3_3", "5_5_21", "5_13_13", "5_10_9", "5_13_6", "5_9_17", "5_5_9", "5_10_20", "5_12_17", "5_11_18", "5_6_16", "5_9_11", "5_2_18", "5_2_9", "5_8_18", "5_13_18", "5_9_5", "5_3_18", "5_2_13", "5_12_3", "5_10_16", "5_3_13", "5_2_17", "5_10_8", "5_5_8", "5_13_2", "5_13_14", "5_13_5", "5_12_6", "5_6_1", "5_11_4", "5_3_5", "5_12_0", "5_6_2", "5_12_18", "5_6_13", "5_13_1", "5_9_13", "5_10_24", "5_3_6", "5_5_23", "5_12_15", "5_12_7", "5_13_9", "5_2_6", "5_10_14", "5_8_11", "5_10_15", "5_6_3", "5_9_0", "5_9_22", "5_8_10", "5_8_12", "5_8_19", "5_2_3", "5_2_2", "5_13_8", "5_10_2", "5_9_6", "5_8_4", "5_10_6", "5_6_14", "5_9_10", "5_3_21", "5_11_12", "5_8_5", "5_10_22", "5_13_12", "5_8_23", "5_5_4", "5_10_13", "5_5_19", "5_8_8", "5_6_17", "5_3_17", "5_5_11", "6_11_2", "6_11_21", "6_3_5", "6_6_19", "6_12_4", "6_3_24", "6_13_0", "6_6_16", "6_3_18", "6_3_13", "6_9_20", "6_9_24", "6_9_21", "6_2_12", "6_13_20", "6_6_22", "6_9_13", "6_8_3", "6_2_20", "6_2_4", "6_10_0", "6_9_6", "6_10_13", "6_3_11", "6_11_11", "6_8_13", "6_8_8", "6_9_9", "6_12_10", "6_10_7", "6_12_16", "6_3_14", "6_8_17", "6_6_11", "6_9_2", "6_8_0", "6_10_1", "6_5_5", "6_3_4", "6_2_16", "6_8_2", "6_6_2", "6_10_8", "6_9_3", "6_9_14", "6_10_12", "6_2_1", "6_9_0", "6_2_15", "6_9_15", "6_10_14", "6_5_13", "6_3_9", "6_12_24", "6_3_2", "6_6_10", "6_5_20", "6_9_22", "6_6_9", "6_9_1", "6_6_24", "6_13_7", "6_5_2", "6_10_20", "6_12_9", "6_10_3", "6_9_8", "6_2_8", "6_12_23", "6_10_5", "6_8_15", "6_13_12", "6_9_16", "6_10_10", "6_5_14", "6_13_1", "6_11_3", "6_13_13", "6_3_20", "6_9_11", "6_5_12", "6_12_11", "6_9_18", "6_6_21", "6_8_12", "6_5_10", "6_8_23", "6_3_6", "6_9_12", "6_8_9", "6_5_1", "6_6_20", "6_11_19", "6_10_19", "6_8_20", "6_5_11", "6_5_8", "6_9_4", "6_2_21", "6_11_1", "6_6_13", "6_10_15", "6_2_7", "6_8_14", "6_3_3", "6_9_10", "6_5_0", "6_6_5", "6_3_0", "6_10_4", "6_6_23", "6_12_12", "6_3_15", "6_8_11", "6_10_21", "6_3_10", "6_13_5", "6_6_15", "6_12_6", "6_10_9", "6_11_23", "6_5_4", "6_6_7", "6_8_21", "6_12_21", "6_8_18", "7_12_6", "7_8_1", "7_11_17", "7_2_3", "7_10_14", "7_6_9", "7_5_9", "7_6_23", "7_5_14", "7_3_13", "7_2_5", "7_13_20", "7_10_24", "7_13_21", "7_5_20", "7_8_20", "7_12_18", "7_8_18", "7_11_16", "7_3_17", "7_10_21", "7_5_13", "7_2_13", "7_11_3", "7_5_4", "7_2_20", "7_9_10", "7_6_24", "7_13_15", "7_2_21", "7_2_4", "7_8_7", "7_13_9", "7_8_4", "7_5_5", "7_13_8", "7_6_2", "7_13_10", "7_9_13", "7_8_23", "7_9_11", "7_8_11", "7_9_19", "7_3_12", "7_10_6", "7_8_10", "7_3_3", "7_11_15", "7_3_7", "7_6_14", "7_2_9", "7_11_13", "7_12_7", "7_3_10", "7_5_8", "7_2_24", "7_10_12", "7_11_7", "7_11_10", "7_13_16", "7_10_1", "7_6_15", "7_8_5", "7_3_20", "7_6_0", "7_9_8", "7_12_17", "7_12_24", "7_2_0", "7_10_20", "7_10_2", "7_9_1", "7_11_5", "7_12_16", "7_11_12", "7_9_9", "7_3_2", "7_6_21", "7_13_5", "7_10_23", "7_11_14", "7_9_3", "7_2_15", "7_6_7", "7_2_17", "7_6_3", "7_5_22", "7_9_21", "7_12_23", "7_10_18", "7_3_14", "7_8_2", "7_5_23", "7_12_5", "7_13_23", "7_2_16", "7_12_22", "7_9_4", "7_13_17", "7_6_4", "7_10_11", "7_10_7", "7_3_1", "7_5_15", "7_9_14", "7_2_1", "7_9_12", "7_5_11", "7_11_9", "7_6_20", "7_6_19", "7_12_0", "7_13_2", "7_6_1", "7_13_11", "7_11_21", "7_8_19", "7_11_23", "7_8_13", "7_12_20", "7_2_22", "7_11_22", "7_10_22", "7_9_0", "7_3_19", "7_3_5", "8_11_7", "8_3_20", "8_13_6", "8_2_5", "8_13_16", "8_13_2", "8_10_14", "8_13_20", "8_8_0", "8_2_20", "8_2_24", "8_2_7", "8_11_2", "8_2_12", "8_2_14", "8_2_17", "8_10_11", "8_6_22", "8_10_19", "8_12_17", "8_2_11", "8_9_9", "8_9_0", "8_11_11", "8_13_0", "8_11_6", "8_5_24", "8_10_5", "8_5_9", "8_13_4", "8_3_2", "8_6_16", "8_13_5", "8_8_11", "8_5_19", "8_10_7", "8_12_13", "8_13_14", "8_8_3", "8_12_18", "8_3_16", "8_12_23", "8_10_8", "8_11_17", "8_8_16", "8_3_9", "8_8_22", "8_11_3", "8_10_24", "8_3_23", "8_6_13", "8_5_17", "8_2_4", "8_13_21", "8_12_1", "8_10_17", "8_6_2", "8_6_7", "8_11_24", "8_3_10", "8_12_20", "8_3_3", "8_12_14", "8_5_6", "8_8_18", "8_11_13", "8_2_23", "8_2_1", "8_12_12", "8_8_6", "8_12_16", "8_5_4", "8_3_6", "8_12_2", "8_10_3", "8_5_23", "8_12_4", "8_11_12", "8_6_5", "8_13_24", "8_3_19", "8_10_18", "8_12_21", "8_9_16", "8_9_3", "8_3_22", "8_8_9", "8_3_17", "8_9_7", "8_5_22", "8_8_10", "8_9_11", "8_12_3", "8_9_22", "8_2_13", "8_6_23", "8_6_20", "8_6_18", "8_12_7", "8_10_0", "8_8_7", "8_12_19", "8_8_20", "8_12_9", "8_9_14", "8_12_15", "8_3_4", "8_11_1", "8_6_14", "8_6_15", "8_9_13", "8_10_10", "8_3_12", "8_5_12", "8_2_21", "8_13_7", "8_10_20", "8_6_3", "8_8_8", "8_6_12", "8_5_3", "8_6_8", "8_11_0", "8_8_19", "8_11_19", "8_9_5", "9_8_3", "9_9_18", "9_8_13", "9_9_16", "9_8_5", "9_11_23", "9_6_1", "9_9_17", "9_3_20", "9_2_23", "9_9_10", "9_12_21", "9_5_21", "9_10_10", "9_8_22", "9_6_7", "9_3_23", "9_9_21", "9_11_8", "9_2_17", "9_9_12", "9_11_15", "9_11_6", "9_2_21", "9_2_0", "9_13_1", "9_9_20", "9_5_22", "9_12_20", "9_2_14", "9_10_23", "9_5_7", "9_12_0", "9_3_11", "9_10_5", "9_2_16", "9_5_12", "9_8_16", "9_13_9", "9_12_15", "9_12_24", "9_5_14", "9_9_13", "9_13_0", "9_6_4", "9_9_23", "9_13_19", "9_13_12", "9_9_4", "9_9_5", "9_8_12", "9_13_23", "9_5_9", "9_8_14", "9_13_4", "9_3_10", "9_5_11", "9_3_9", "9_3_3", "9_11_14", "9_2_22", "9_9_15", "9_13_3", "9_10_19", "9_11_12", "9_12_22", "9_8_1", "9_11_5", "9_10_16", "9_3_21", "9_10_18", "9_12_17", "9_10_15", "9_3_7", "9_5_18", "9_5_19", "9_2_4", "9_12_3", "9_3_17", "9_2_5", "9_13_24", "9_6_17", "9_6_11", "9_11_21", "9_10_0", "9_11_18", "9_10_11", "9_8_24", "9_12_18", "9_10_2", "9_2_8", "9_9_3", "9_11_9", "9_5_23", "9_5_15", "9_12_6", "9_9_22", "9_2_10", "9_2_3", "9_10_12", "9_11_17", "9_3_14", "9_13_14", "9_8_0", "9_3_8", "9_5_1", "9_13_17", "9_12_12", "9_6_19", "9_3_15", "9_8_11", "9_3_4", "9_10_21", "9_6_21", "9_10_9", "9_8_2", "9_6_10", "9_8_4", "9_2_13", "9_3_1", "9_6_20", "9_10_14", "9_8_7", "9_3_13", "9_5_0", "9_5_17", "10_10_16", "10_10_7", "10_3_23", "10_5_0", "10_6_16", "10_10_11", "10_6_2", "10_5_9", "10_12_7", "10_10_19", "10_13_15", "10_10_3", "10_6_7", "10_11_17", "10_9_0", "10_6_5", "10_3_11", "10_13_0", "10_13_2", "10_10_13", "10_11_20", "10_3_19", "10_13_14", "10_5_15", "10_9_19", "10_9_22", "10_11_3", "10_2_1", "10_6_13", "10_12_11", "10_9_20", "10_8_6", "10_10_20", "10_8_2", "10_8_19", "10_9_14", "10_2_14", "10_13_18", "10_9_16", "10_2_13", "10_9_7", "10_3_1", "10_9_4", "10_2_15", "10_9_6", "10_8_4", "10_12_13", "10_13_10", "10_12_1", "10_12_2", "10_13_13", "10_8_0", "10_10_6", "10_12_20", "10_5_16", "10_2_20", "10_6_8", "10_12_15", "10_2_2", "10_9_1", "10_13_19", "10_9_15", "10_8_20", "10_8_22", "10_2_8", "10_2_7", "10_11_23", "10_10_1", "10_9_17", "10_10_5", "10_10_18", "10_11_8", "10_6_17", "10_11_12", "10_3_15", "10_13_20", "10_3_17", "10_11_0", "10_2_24", "10_12_12", "10_12_10", "10_13_4", "10_8_16", "10_8_11", "10_8_3", "10_6_18", "10_8_10", "10_5_14", "10_12_17", "10_6_23", "10_3_13", "10_13_12", "10_5_19", "10_6_20", "10_11_21", "10_11_7", "10_12_4", "10_13_5", "10_2_12", "10_11_6", "10_10_15", "10_2_19", "10_10_22", "10_13_9", "10_5_5", "10_5_1", "10_5_24", "10_2_9", "10_6_22", "10_12_16", "10_5_6", "10_13_11", "10_13_21", "10_10_0", "10_9_10", "10_10_8", "10_9_3", "10_5_12", "10_12_19", "10_8_9", "10_3_12", "10_11_11", "10_11_16", "10_5_4", "10_6_19", "10_3_7"],
            "eval": ["0_12_20", "0_10_18", "0_6_8", "0_3_0", "0_12_22", "0_13_10", "0_12_3", "0_5_14", "0_10_15", "0_2_5", "0_3_5", "0_3_18", "0_12_21", "0_5_21", "0_10_20", "0_8_22", "0_8_11", "0_13_6", "0_9_0", "0_3_7", "0_8_8", "0_5_10", "0_10_6", "0_9_20", "0_6_20", "0_13_15", "0_6_2", "0_9_13", "0_13_17", "0_12_23", "0_2_4", "0_12_10", "0_11_12", "0_13_21", "0_13_9", "0_12_6", "0_5_12", "0_9_16", "0_2_18", "0_13_4", "0_6_1", "0_10_12", "0_11_1", "0_11_6", "0_13_16", "0_2_13", "0_2_12", "0_5_16", "0_8_10", "0_13_19", "0_12_2", "0_12_17", "0_9_19", "0_6_3", "0_13_2", "0_9_6", "0_5_13", "0_12_9", "0_6_21", "0_5_19", "0_10_9", "0_9_23", "0_2_15", "0_9_22", "0_3_15", "0_11_21", "0_8_18", "0_12_12", "0_12_8", "0_5_22", "0_10_23", "0_5_17", "0_13_24", "0_6_12", "0_6_13", "0_10_24", "0_8_0", "0_5_24", "0_12_0", "0_8_6", "0_8_15", "0_5_11", "0_2_20", "0_8_2", "0_6_16", "0_9_21", "0_6_14", "0_3_21", "0_10_4", "0_13_8", "0_10_22", "0_9_18", "0_6_24", "0_13_1", "0_11_10", "0_5_4", "0_8_1", "0_13_11", "0_10_19", "0_2_23", "0_13_3", "0_11_8", "0_8_9", "0_12_15", "0_11_15", "0_3_24", "0_8_17", "0_3_10", "0_5_1", "0_2_21", "0_10_17", "0_2_8", "0_9_8", "0_13_7", "0_6_9", "0_5_20", "0_12_24", "0_12_11", "0_13_0", "0_9_11", "0_5_23", "0_11_11", "0_6_23", "0_13_14", "1_6_23", "1_9_5", "1_11_16", "1_10_16", "1_13_22", "1_3_18", "1_6_19", "1_8_23", "1_9_20", "1_8_9", "1_8_24", "1_9_15", "1_12_9", "1_5_2", "1_5_24", "1_6_17", "1_10_3", "1_10_22", "1_6_7", "1_12_1", "1_9_10", "1_12_10", "1_6_5", "1_8_21", "1_5_17", "1_3_20", "1_13_23", "1_2_15", "1_8_7", "1_11_1", "1_2_7", "1_11_23", "1_13_8", "1_6_13", "1_2_21", "1_3_13", "1_13_15", "1_6_18", "1_9_9", "1_2_19", "1_12_17", "1_11_11", "1_5_20", "1_5_16", "1_12_6", "1_3_7", "1_2_20", "1_11_7", "1_6_1", "1_9_22", "1_10_0", "1_6_22", "1_9_14", "1_2_3", "1_6_6", "1_8_17", "1_3_19", "1_11_0", "1_8_18", "1_10_15", "1_9_8", "1_9_18", "1_5_0", "1_10_23", "1_5_23", "1_10_13", "1_10_7", "1_6_8", "1_11_6", "1_8_0", "1_8_6", "1_13_17", "1_12_4", "1_13_13", "1_9_0", "1_6_20", "1_5_4", "1_3_3", "1_3_23", "1_12_20", "1_10_4", "1_11_2", "1_8_20", "1_12_3", "1_8_8", "1_12_8", "1_12_0", "1_5_11", "1_13_12", "1_9_11", "1_2_1", "1_12_13", "1_10_17", "1_3_22", "1_6_9", "1_3_12", "1_12_16", "1_8_15", "1_12_23", "1_5_5", "1_11_5", "1_11_20", "1_13_16", "1_8_4", "1_5_13", "1_13_18", "1_8_13", "1_8_1", "1_10_9", "1_13_4", "1_6_12", "1_5_7", "1_6_21", "1_9_2", "1_3_11", "1_5_3", "1_10_19", "1_9_4", "1_9_3", "1_13_0", "1_12_19", "1_10_11", "1_11_13", "1_12_21", "2_6_24", "2_10_13", "2_12_18", "2_5_2", "2_2_10", "2_12_4", "2_2_15", "2_10_14", "2_9_11", "2_9_19", "2_8_5", "2_9_14", "2_10_6", "2_3_18", "2_13_12", "2_2_0", "2_11_19", "2_12_21", "2_5_0", "2_8_19", "2_9_18", "2_13_5", "2_3_12", "2_3_6", "2_12_6", "2_9_21", "2_9_4", "2_3_14", "2_11_10", "2_11_1", "2_9_6", "2_2_17", "2_5_16", "2_6_9", "2_2_16", "2_6_11", "2_9_9", "2_9_12", "2_8_15", "2_2_1", "2_6_10", "2_13_16", "2_12_1", "2_9_13", "2_10_3", "2_6_4", "2_6_1", "2_11_12", "2_12_12", "2_11_7", "2_3_17", "2_6_17", "2_9_0", "2_2_8", "2_3_24", "2_9_3", "2_8_6", "2_13_0", "2_9_2", "2_5_10", "2_13_4", "2_3_10", "2_6_3", "2_9_1", "2_3_11", "2_9_10", "2_13_14", "2_5_7", "2_6_19", "2_13_19", "2_10_1", "2_2_19", "2_11_20", "2_13_24", "2_10_4", "2_6_18", "2_9_15", "2_6_15", "2_5_22", "2_13_3", "2_10_16", "2_11_11", "2_8_17", "2_10_2", "2_6_0", "2_3_21", "2_3_15", "2_13_8", "2_6_22", "2_10_9", "2_12_16", "2_5_18", "2_8_10", "2_10_15", "2_8_22", "2_13_13", "2_12_8", "2_13_15", "2_11_22", "2_2_4", "2_11_13", "2_3_4", "2_5_4", "2_11_4", "2_8_14", "2_2_6", "2_9_16", "2_11_23", "2_12_13", "2_3_9", "2_8_20", "2_6_20", "2_11_15", "2_6_21", "2_13_22", "2_5_9", "2_10_8", "2_12_0", "2_11_3", "2_2_24", "2_5_24", "2_13_23", "2_8_8", "2_2_21", "3_13_24", "3_2_17", "3_11_3", "3_5_9", "3_2_0", "3_3_23", "3_10_23", "3_3_6", "3_10_9", "3_8_18", "3_5_12", "3_11_17", "3_8_5", "3_2_15", "3_6_14", "3_9_12", "3_9_9", "3_13_10", "3_11_14", "3_8_3", "3_2_8", "3_11_8", "3_8_1", "3_3_15", "3_5_0", "3_13_0", "3_8_2", "3_6_7", "3_13_3", "3_11_1", "3_2_20", "3_8_16", "3_5_3", "3_2_13", "3_2_7", "3_2_4", "3_5_18", "3_6_3", "3_3_7", "3_9_18", "3_6_15", "3_9_15", "3_2_24", "3_11_22", "3_2_18", "3_10_12", "3_8_0", "3_8_13", "3_8_21", "3_9_6", "3_3_21", "3_12_13", "3_3_1", "3_8_15", "3_11_4", "3_3_0", "3_13_19", "3_11_16", "3_6_9", "3_2_19", "3_9_17", "3_13_11", "3_5_20", "3_13_7", "3_3_3", "3_13_23", "3_9_16", "3_9_8", "3_5_13", "3_5_23", "3_10_15", "3_8_19", "3_12_5", "3_12_4", "3_9_19", "3_6_20", "3_11_21", "3_6_1", "3_11_12", "3_2_22", "3_2_5", "3_6_13", "3_12_6", "3_2_16", "3_9_0", "3_8_23", "3_9_10", "3_6_0", "3_11_15", "3_12_9", "3_5_1", "3_13_20", "3_13_8", "3_13_6", "3_10_5", "3_10_24", "3_8_22", "3_6_10", "3_8_6", "3_8_7", "3_9_5", "3_11_11", "3_2_14", "3_6_2", "3_5_10", "3_11_23", "3_8_11", "3_10_0", "3_2_1", "3_11_24", "3_12_23", "3_12_17", "3_12_0", "3_12_24", "3_12_2", "3_9_2", "3_3_9", "3_5_7", "3_13_16", "3_10_4", "3_12_8", "3_6_4", "3_2_6", "3_3_24", "4_12_3", "4_6_12", "4_10_13", "4_3_21", "4_6_21", "4_3_15", "4_11_16", "4_13_5", "4_3_4", "4_10_2", "4_6_7", "4_9_11", "4_10_8", "4_2_20", "4_3_6", "4_3_2", "4_11_11", "4_8_16", "4_2_0", "4_2_2", "4_5_6", "4_12_1", "4_6_19", "4_8_0", "4_12_8", "4_13_10", "4_2_21", "4_3_24", "4_12_22", "4_11_8", "4_5_17", "4_13_4", "4_8_1", "4_11_14", "4_11_20", "4_10_21", "4_13_2", "4_13_1", "4_8_12", "4_9_0", "4_12_11", "4_12_2", "4_2_24", "4_3_11", "4_6_16", "4_12_7", "4_6_3", "4_9_13", "4_9_2", "4_13_6", "4_6_9", "4_8_18", "4_12_24", "4_2_11", "4_2_18", "4_9_8", "4_5_22", "4_5_10", "4_11_24", "4_13_22", "4_13_17", "4_8_11", "4_13_14", "4_8_20", "4_11_13", "4_2_1", "4_12_16", "4_11_15", "4_13_0", "4_2_5", "4_12_20", "4_5_5", "4_10_12", "4_8_21", "4_13_13", "4_10_6", "4_9_7", "4_2_19", "4_10_15", "4_10_22", "4_11_1", "4_10_7", "4_12_0", "4_2_12", "4_12_15", "4_9_9", "4_8_17", "4_8_8", "4_12_10", "4_3_23", "4_6_22", "4_8_4", "4_11_10", "4_2_9", "4_5_23", "4_6_4", "4_5_3", "4_13_12", "4_3_10", "4_10_4", "4_12_6", "4_6_13", "4_10_10", "4_9_12", "4_8_6", "4_13_16", "4_2_17", "4_2_10", "4_13_8", "4_11_21", "4_12_12", "4_11_9", "4_8_3", "4_13_18", "4_5_9", "4_5_20", "4_10_14", "4_6_5", "4_10_5", "4_11_2", "4_6_11", "4_5_13", "4_5_21", "4_6_17", "5_8_21", "5_10_19", "5_2_8", "5_5_16", "5_9_21", "5_2_20", "5_6_18", "5_6_8", "5_9_18", "5_12_11", "5_10_17", "5_13_11", "5_8_3", "5_5_13", "5_11_7", "5_12_8", "5_2_0", "5_3_23", "5_8_17", "5_5_6", "5_10_5", "5_8_9", "5_10_21", "5_12_10", "5_3_16", "5_5_14", "5_9_1", "5_11_21", "5_11_16", "5_5_22", "5_2_24", "5_13_15", "5_8_0", "5_10_1", "5_8_14", "5_8_20", "5_12_19", "5_8_6", "5_6_7", "5_5_1", "5_11_14", "5_3_19", "5_11_9", "5_5_24", "5_6_20", "5_8_24", "5_3_12", "5_12_2", "5_5_2", "5_12_16", "5_3_0", "5_11_10", "5_11_20", "5_13_19", "5_5_5", "5_3_15", "5_9_14", "5_9_24", "5_13_24", "5_11_23", "5_13_7", "5_13_23", "5_6_9", "5_6_24", "5_2_12", "5_5_7", "5_3_22", "5_12_12", "5_5_18", "5_12_23", "5_13_22", "5_2_23", "5_11_24", "5_11_17", "5_2_14", "5_3_4", "5_11_6", "5_11_0", "5_6_6", "5_10_4", "5_5_20", "5_11_15", "5_10_3", "5_3_10", "5_11_2", "5_9_4", "5_9_2", "5_3_14", "5_10_7", "5_10_23", "5_11_3", "5_6_4", "5_9_19", "5_9_8", "5_6_22", "5_8_7", "5_6_11", "5_9_9", "5_6_23", "5_9_15", "5_3_1", "5_12_20", "5_5_3", "5_12_21", "5_2_22", "5_13_16", "5_8_15", "5_10_18", "5_13_20", "5_9_12", "5_6_10", "5_10_10", "5_13_10", "5_8_2", "5_13_17", "5_10_12", "5_12_13", "5_3_9", "5_6_12", "5_2_16", "5_11_19", "5_2_5", "5_2_1", "5_11_22", "6_2_11", "6_2_17", "6_11_13", "6_2_24", "6_13_18", "6_5_19", "6_6_14", "6_12_1", "6_13_23", "6_5_15", "6_10_6", "6_12_19", "6_12_22", "6_6_12", "6_6_1", "6_10_17", "6_3_23", "6_13_9", "6_9_5", "6_13_19", "6_6_4", "6_5_22", "6_12_15", "6_5_21", "6_8_1", "6_12_7", "6_2_9", "6_11_16", "6_2_5", "6_2_18", "6_5_7", "6_6_6", "6_11_7", "6_11_6", "6_12_14", "6_11_9", "6_12_20", "6_8_16", "6_10_18", "6_2_19", "6_10_2", "6_2_22", "6_3_22", "6_6_18", "6_11_22", "6_13_22", "6_5_24", "6_3_19", "6_12_5", "6_5_17", "6_8_7", "6_8_19", "6_8_6", "6_13_8", "6_13_2", "6_9_19", "6_11_17", "6_11_4", "6_9_7", "6_13_21", "6_5_6", "6_11_0", "6_5_18", "6_2_10", "6_13_3", "6_13_24", "6_8_10", "6_12_0", "6_10_24", "6_13_11", "6_5_23", "6_2_3", "6_13_16", "6_13_6", "6_3_12", "6_12_17", "6_11_15", "6_11_24", "6_12_2", "6_3_21", "6_12_18", "6_9_23", "6_8_24", "6_2_13", "6_6_17", "6_6_0", "6_6_3", "6_6_8", "6_13_15", "6_3_8", "6_11_10", "6_10_16", "6_8_22", "6_2_2", "6_5_9", "6_8_5", "6_2_23", "6_11_20", "6_10_11", "6_11_14", "6_2_6", "6_2_0", "6_3_1", "6_3_17", "6_13_4", "6_13_10", "6_10_22", "6_11_8", "6_5_16", "6_8_4", "6_13_17", "6_2_14", "6_10_23", "6_5_3", "6_11_5", "6_12_8", "6_11_12", "6_13_14", "6_9_17", "6_11_18", "6_12_3", "6_12_13", "6_3_16", "6_3_7", "7_6_5", "7_12_19", "7_11_18", "7_2_6", "7_9_23", "7_3_21", "7_3_23", "7_12_2", "7_12_14", "7_6_16", "7_12_3", "7_8_9", "7_10_4", "7_2_12", "7_12_8", "7_5_12", "7_9_15", "7_9_16", "7_6_11", "7_9_5", "7_11_4", "7_8_0", "7_5_21", "7_5_19", "7_5_10", "7_9_7", "7_2_7", "7_8_22", "7_9_24", "7_10_5", "7_10_15", "7_5_7", "7_10_13", "7_6_17", "7_2_23", "7_9_20", "7_11_20", "7_13_14", "7_12_12", "7_9_2", "7_3_22", "7_3_18", "7_5_3", "7_9_6", "7_11_0", "7_8_15", "7_6_6", "7_5_1", "7_13_24", "7_3_4", "7_9_17", "7_5_2", "7_12_1", "7_13_0", "7_9_22", "7_12_9", "7_5_17", "7_12_10", "7_13_3", "7_2_10", "7_6_8", "7_2_18", "7_6_18", "7_13_13", "7_11_1", "7_12_21", "7_12_15", "7_13_18", "7_8_21", "7_13_7", "7_3_9", "7_3_8", "7_2_11", "7_8_12", "7_2_2", "7_12_4", "7_6_12", "7_10_16", "7_13_4", "7_5_16", "7_10_9", "7_11_19", "7_3_16", "7_3_15", "7_2_8", "7_11_2", "7_3_24", "7_6_13", "7_10_19", "7_10_0", "7_8_6", "7_10_17", "7_10_3", "7_8_14", "7_13_22", "7_2_19", "7_8_8", "7_3_11", "7_3_6", "7_5_24", "7_5_0", "7_3_0", "7_8_17", "7_12_11", "7_5_18", "7_11_6", "7_8_3", "7_6_22", "7_11_11", "7_9_18", "7_5_6", "7_13_19", "7_10_8", "7_11_24", "7_12_13", "7_8_24", "7_13_6", "7_13_12", "7_8_16", "7_10_10", "7_13_1", "7_6_10", "7_11_8", "7_2_14", "8_11_15", "8_12_10", "8_13_11", "8_10_6", "8_6_21", "8_5_10", "8_9_10", "8_13_17", "8_10_22", "8_9_20", "8_8_12", "8_8_2", "8_5_2", "8_11_23", "8_13_22", "8_3_21", "8_12_8", "8_2_16", "8_3_18", "8_8_17", "8_9_15", "8_2_10", "8_5_1", "8_9_23", "8_9_2", "8_8_1", "8_6_10", "8_5_11", "8_2_3", "8_5_7", "8_2_6", "8_11_8", "8_8_4", "8_6_9", "8_5_5", "8_9_8", "8_10_1", "8_13_1", "8_9_1", "8_3_0", "8_3_13", "8_11_21", "8_8_5", "8_13_13", "8_6_24", "8_6_11", "8_2_9", "8_5_15", "8_5_13", "8_2_8", "8_13_15", "8_10_12", "8_11_10", "8_10_2", "8_5_0", "8_9_24", "8_9_21", "8_3_15", "8_2_19", "8_2_22", "8_10_13", "8_5_8", "8_5_16", "8_9_6", "8_9_19", "8_6_17", "8_3_7", "8_11_5", "8_6_19", "8_12_0", "8_2_2", "8_2_0", "8_10_4", "8_8_23", "8_10_16", "8_11_18", "8_13_3", "8_11_9", "8_3_14", "8_3_5", "8_13_8", "8_5_18", "8_10_23", "8_2_18", "8_13_23", "8_9_18", "8_13_12", "8_8_14", "8_12_24", "8_12_5", "8_9_12", "8_3_1", "8_11_16", "8_6_6", "8_12_11", "8_8_15", "8_8_24", "8_6_4", "8_2_15", "8_11_20", "8_11_14", "8_13_19", "8_10_15", "8_10_9", "8_13_10", "8_12_22", "8_13_9", "8_11_22", "8_5_20", "8_11_4", "8_3_24", "8_8_13", "8_9_17", "8_6_0", "8_12_6", "8_3_11", "8_6_1", "8_9_4", "8_13_18", "8_5_14", "8_10_21", "8_8_21", "8_5_21", "8_3_8", "9_2_1", "9_9_14", "9_10_1", "9_9_1", "9_13_2", "9_3_0", "9_11_24", "9_11_0", "9_12_23", "9_2_24", "9_6_8", "9_11_3", "9_6_9", "9_2_18", "9_6_24", "9_11_16", "9_2_12", "9_2_15", "9_2_20", "9_13_13", "9_13_6", "9_6_14", "9_2_9", "9_2_11", "9_6_22", "9_3_16", "9_9_2", "9_9_19", "9_13_16", "9_9_8", "9_12_4", "9_13_7", "9_12_7", "9_13_10", "9_3_19", "9_8_10", "9_10_13", "9_13_15", "9_8_19", "9_12_1", "9_11_4", "9_12_10", "9_13_5", "9_10_3", "9_12_19", "9_3_18", "9_5_16", "9_3_22", "9_11_22", "9_5_8", "9_5_20", "9_13_22", "9_10_17", "9_12_16", "9_10_24", "9_13_8", "9_8_15", "9_13_18", "9_3_24", "9_2_2", "9_8_20", "9_8_17", "9_10_20", "9_11_2", "9_9_6", "9_8_9", "9_8_23", "9_5_10", "9_8_18", "9_12_9", "9_10_7", "9_3_2", "9_13_11", "9_6_3", "9_9_24", "9_6_18", "9_6_0", "9_10_8", "9_9_9", "9_6_6", "9_11_7", "9_6_5", "9_11_13", "9_13_20", "9_11_10", "9_9_11", "9_10_22", "9_11_19", "9_6_16", "9_12_13", "9_6_12", "9_12_5", "9_9_7", "9_5_6", "9_5_5", "9_12_14", "9_9_0", "9_6_23", "9_10_4", "9_3_5", "9_6_2", "9_3_6", "9_6_13", "9_11_11", "9_5_13", "9_2_19", "9_12_11", "9_11_20", "9_8_21", "9_11_1", "9_5_4", "9_2_6", "9_5_3", "9_5_2", "9_8_8", "9_6_15", "9_2_7", "9_5_24", "9_12_8", "9_13_21", "9_3_12", "9_8_6", "9_10_6", "9_12_2", "10_13_24", "10_13_16", "10_12_8", "10_3_2", "10_12_18", "10_12_3", "10_9_2", "10_10_4", "10_9_9", "10_9_5", "10_8_18", "10_2_6", "10_13_1", "10_9_21", "10_11_1", "10_10_17", "10_13_17", "10_2_17", "10_13_6", "10_12_24", "10_2_5", "10_9_11", "10_13_7", "10_9_8", "10_10_2", "10_5_8", "10_12_23", "10_6_11", "10_3_24", "10_5_3", "10_6_15", "10_11_22", "10_9_24", "10_10_23", "10_9_18", "10_8_23", "10_6_6", "10_2_21", "10_10_21", "10_8_12", "10_2_3", "10_8_7", "10_6_9", "10_8_14", "10_5_17", "10_3_16", "10_11_5", "10_8_8", "10_3_21", "10_3_18", "10_3_10", "10_2_10", "10_8_15", "10_8_5", "10_6_4", "10_10_14", "10_2_4", "10_3_0", "10_11_18", "10_9_12", "10_6_24", "10_12_5", "10_13_3", "10_3_8", "10_3_9", "10_12_0", "10_6_10", "10_2_23", "10_9_23", "10_11_10", "10_11_9", "10_11_14", "10_2_16", "10_13_8", "10_3_22", "10_5_13", "10_5_22", "10_12_9", "10_3_6", "10_5_21", "10_5_2", "10_10_9", "10_11_19", "10_5_20", "10_3_14", "10_5_10", "10_11_15", "10_10_10", "10_8_21", "10_12_22", "10_11_24", "10_6_14", "10_12_14", "10_5_7", "10_11_13", "10_8_24", "10_6_12", "10_6_21", "10_6_1", "10_5_23", "10_6_0", "10_5_18", "10_11_4", "10_12_21", "10_3_3", "10_2_11", "10_11_2", "10_5_11", "10_6_3", "10_10_24", "10_9_13", "10_8_13", "10_8_17", "10_3_20", "10_13_22", "10_2_22", "10_13_23", "10_8_1", "10_10_12", "10_2_18", "10_3_4", "10_2_0", "10_12_6", "10_3_5"]}
trainPart = use_file['train']
testPart = use_file['eval']

################### Define data

def shuffle_batch_seq(batch_num, seq_len, list_data):

    count = 0
    
    while(count < batch_num): 
        shuffle(list_data)
        list_data = list_data[:10]
        idir = dest + list_data[0]+ str('.h5')
        f = h5py.File(idir,'r')             # read the .h5 files and shuffle them
        channel_data = f['dataset'][()]
        label = f['label'][()]
        f.close()
    
        for data in list_data[1:5]:
            idir = dest + data + str('.h5')
            f = h5py.File(idir,'r')       
            temp = f['dataset'][()]
            label_ = f['label'][()]            
            
            channel_data = np.append(channel_data, temp, axis=0)
            label = np.append(label, label_ , axis=0)
            f.close()
#        rand_start = random.randint(0, len(channel_data)- seq_len) # randomly choose the start
#        channel_data = channel_data[rand_start: rand_start+ seq_len, :, :]
#        label = label[rand_start: rand_start+ seq_len, :]
        channel_data = channel_data[0:seq_len, :, :]
        label = label[0:seq_len, :]
        channel_data = channel_data[..., np.newaxis]
        label = label[..., np.newaxis]
       
        if count == 0: # append data
            batch_xs = channel_data
            batch_ys = label
        else:
            batch_xs = np.concatenate((batch_xs, channel_data), axis=3)
            batch_ys = np.concatenate((batch_ys, label), axis=2)
        
        count += 1 
    batch_xs = np.transpose(batch_xs, (3,0,1,2))
    batch_ys = np.transpose(batch_ys, (2,0,1))
    
    return batch_xs, batch_ys

#a, b = shuffle_batch_seq(5, 200, train)

