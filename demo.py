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
from perceptron import Perceptron
from pytorch import PyTorchNN
from nn_2 import NeuralNetwork as NeuralNetwork2

load_dotenv()

def create_nn(classifier, input_size, num_classes, percentage, run):
    filename = ""
    if classifier == 0: # digits dataset
        filename = f"digitWeights/{int(percentage*100)}/run{run}.npz"
    else:
        filename = f"faceWeights/{int(percentage*100)}/run{run}.npz"
    path = os.path.join(os.getenv("neural_net_weights_path"), filename)
    weights = np.load(path)
    weights_1 = weights['weights_1']
    weights_2 = weights['weights_2']
    weights_3 = weights['weights_3']

    nn = NeuralNetwork(input_size, 20, 20, num_classes)
    nn.set_weights(weights_1, weights_2, weights_3)
    return nn

def create_perceptron(classifier, input_size, num_classes, percentage, run):
    filename = ""
    if classifier == 0: # digits dataset
        filename = f"digitWeights/{int(percentage*100)}/run{run}.npz"
    else:
        filename = f"faceWeights/{int(percentage*100)}/run{run}.npz"
    path = os.path.join(os.getenv("perceptron_weights_path"), filename)
    weights = np.load(path)
    weights = weights['weights']
    perceptron = Perceptron(input_size, num_classes)
    perceptron.set_weights(weights)
    return perceptron

def create_pytorch_nn(classifier, input_size, num_classes, percentage, run):
    filename = ""
    hidden_size = 100
    if classifier == 0: # digits dataset
        filename = f"digitWeights/{int(percentage*100)}/run{run}.pth"
    else:
        filename = f"faceWeights/{int(percentage*100)}/run{run}.pth"
    
    path = os.path.join(os.getenv("pytorch_weights_path"), filename)

    model = PyTorchNN(input_size, hidden_size, num_classes)

    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def create_nn2(classifier, input_size, num_classes, percentage, run):
    filename = ""
    hidden_size_1 = 100
    hidden_size_2 = 100
    if classifier == 0: # digits dataset
        filename = f"digitWeights/{int(percentage*100)}/run{run}.npz"
    else:
        filename = f"faceWeights/{int(percentage*100)}/run{run}.npz"

    path = os.path.join(os.getenv("neural_net_weights_2_path"), filename)

    weights_biases = np.load(path)
    weights_1 = weights_biases['weights_1']
    weights_2 = weights_biases['weights_2']
    weights_3 = weights_biases['weights_3']
    bias_1 = weights_biases['bias_1']
    bias_2 = weights_biases['bias_2']
    bias_3 = weights_biases['bias_3']

    nn2 = NeuralNetwork2(input_size, hidden_size_1, hidden_size_2, num_classes)
    nn2.set_weights_biases(weights_1, weights_2, weights_3, bias_1, bias_2, bias_3)
    return nn2

def test_nn(classifier, X_test, y_test, num_classes, percentage, run):
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    nn = create_nn(classifier, X_test.shape[1], num_classes, percentage, run)
    return nn.test(X_test, y_test)

def test_nn_individual(classifier, X, num_classes, percentage, run):
    X = X.flatten()
    nn = create_nn(classifier, len(X), num_classes, percentage, run)
    nn.forward(X)
    return nn.predict()

def test_perceptron(classifier, X_test, y_test, num_classes, percentage, run):
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    perceptron = create_perceptron(classifier, X_test.shape[1], num_classes, percentage, run)
    return perceptron.test(X_test, y_test)

def test_perceptron_individual(classifier, X, num_classes, percentage, run):
    X = X.flatten()
    perceptron = create_perceptron(classifier, len(X), num_classes, percentage, run)
    augmented_x = np.insert(X, 0, 1)
    return perceptron.predict(augmented_x)

def test_pytorch_nn(classifier, X_test, y_test, num_classes, percentage, run):
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    model = create_pytorch_nn(classifier, X_test.shape[1], num_classes, percentage, run)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return model.test_model(X_test_tensor, y_test_tensor)

def test_pytorch_nn_individual(classifier, X, num_classes, percentage, run):
    X_flat = X.flatten()
    model = create_pytorch_nn(classifier, len(X_flat), num_classes, percentage, run)
    input_tensor = torch.tensor(X_flat, dtype=torch.float32).view(1, -1) # Use view(1, -1)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

def test_nn_2(classifier, X_test, y_test, num_classes, percentage, run):
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    nn2 = create_nn2(classifier, X_test.shape[1], num_classes, percentage, run)
    return nn2.test(X_test, y_test)

def test_nn_2_individual(classifier, X, num_classes, percentage, run):
    X_flat = X.flatten()
    nn2 = create_nn2(classifier, len(X_flat), num_classes, percentage, run)
    if nn2 is None:
        print("Skipping NN_2 individual prediction due to model loading failure.")
        return None

    input_reshaped = X_flat.reshape(1, -1)

    probs = nn2.forward(input_reshaped)

    prediction = np.argmax(probs, axis=1)

    return prediction.item()

if __name__ == "__main__":
    digits_X_train, digits_y_train, digits_X_test, digits_y_test, face_X_train, face_y_train, face_X_test, face_y_test = get_data()
    digits_X_train = np.array(digits_X_train)
    digits_y_train = np.array(digits_y_train)
    face_X_train = np.array(face_X_train)
    face_y_train = np.array(face_y_train)
    digits_X_test = np.array(digits_X_test)
    digits_y_test = np.array(digits_y_test)
    face_X_test = np.array(face_X_test)
    face_y_test = np.array(face_y_test)

    test_nn(0, digits_X_test, digits_y_test, 10, 1, 0)
    test_nn(1, face_X_test, face_y_test, 1, 1, 0)

    print(test_nn_individual(0, digits_X_test[0], 10, 1, 0))
    print(digits_y_test[0])
    print()
    print(test_nn_individual(1, face_X_test[0], 1, 1, 0))
    print(face_y_test[0])
    print("========================================")

    test_perceptron(0, digits_X_test, digits_y_test, 10, 1, 0)
    test_perceptron(1, face_X_test, face_y_test, 1, 1, 0)

    print(test_perceptron_individual(0, digits_X_test[0], 10, 1, 0))
    print(digits_y_test[0])
    print()
    print(test_perceptron_individual(1, face_X_test[0], 1, 1, 0))
    print(face_y_test[0])
    print("========================================")

    test_pytorch_nn(0, digits_X_test, digits_y_test, 10, 1, 0)
    test_pytorch_nn(1, face_X_test, face_y_test, 2, 1, 0)
    print(test_pytorch_nn_individual(0, digits_X_test[0], 10, 1, 0))
    print(digits_y_test[0])
    print()
    print(test_pytorch_nn_individual(1, face_X_test[0], 2, 1, 0))
    print(face_y_test[0])
    print("========================================")

    test_nn_2(0, digits_X_test, digits_y_test, 10, 1, 0)
    test_nn_2(1, face_X_test, face_y_test, 2, 1, 0) # NN_2 face trained for 2 classes
    print(test_nn_2_individual(0, digits_X_test[0], 10, 1, 0))
    print(digits_y_test[0])
    print()
    print(test_nn_2_individual(1, face_X_test[0], 2, 1, 0)) # NN_2 face trained for 2 classes
    print(face_y_test[0])
    print("========================================")