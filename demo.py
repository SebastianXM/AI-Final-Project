import time
import torch 
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from data_processing import get_data
from nn import NeuralNetwork

load_dotenv()

def test_nn(nn, classifier, X_test, y_test, num_classes, percentage, run):
    pass

def test_nn_individual(nn, classifier, X, y, num_classes, percentage, run):
    pass

def test_perceptron(classifier, X_test, y_test, num_classes, percentage, run):
    pass 

def test_perceptron_individual(classifier, X, y, num_classes, percentage, run):
    pass 
