#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:33:02 2019

@author: ys3276
"""
import numpy as np

def arrangeShape(feats, target, sequence_length):
    X = []
    for set_inputs_i in range(len(feats) - sequence_length + 1):
        set_feats = feats[set_inputs_i:set_inputs_i+sequence_length, :]
        X.append(set_feats)
    X = np.array(X, dtype=np.float32)
    Y = target[int(sequence_length/2):len(target)-int((sequence_length-1)/2), :]
    return X, Y
