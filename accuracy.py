#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:31:11 2019

@author: ys3276
"""
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def getAccuracy(predicted, actual, threshold):
    predictedBool = predicted > threshold
    predicted_1d = np.reshape(predictedBool,(-1))
    actual_1d = np.reshape(actual,(-1))
    accuracy = accuracy_score(actual_1d, predicted_1d)
    f1 = f1_score(actual_1d, predicted_1d)
    return accuracy, f1
