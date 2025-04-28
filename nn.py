import time
import random
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from data_processing import get_data

load_dotenv()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = num_classes

        self.input_layer = np.zeros(self.input_size+1)
        self.hidden_layer_1 = np.zeros(self.hidden_size_1+1)
        self.hidden_layer_2 = np.zeros(self.hidden_size_2+1)
        self.output_layer = np.zeros(self.output_size)

        self.input_layer[0] = 1
        self.hidden_layer_1[0] = 1
        self.hidden_layer_2[0] = 1

        self.weights_1 = np.random.rand(self.hidden_size_1, self.input_size+1) * 0.01
        self.weights_2 = np.random.rand(self.hidden_size_2, self.hidden_size_1+1) * 0.01
        self.weights_3 = np.random.rand(self.output_size, self.hidden_size_2+1) * 0.01

    def forward(self, x):
        # x is a 1D array of shape (input_size)
        self.input_layer[1:] = x
        self.hidden_layer_1[1:] = sigmoid(np.matmul(self.weights_1, self.input_layer))
        self.hidden_layer_2[1:] = sigmoid(np.matmul(self.weights_2, self.hidden_layer_1))
        self.output_layer = sigmoid(np.matmul(self.weights_3, self.hidden_layer_2))
        pass

    def predict(self):
        if len(self.output_layer) == 1:
            return 1 if self.output_layer[0] >= 0.5 else 0
        else:
            return np.argmax(self.output_layer)

    def cost_function(self, y, y_hat):
        # return np.mean(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)) # have to check the loss function
        pass

    def back_propogation(self, x, y):
        pass

    def train(self, X_train, y_train, epochs, learning_rate):
        for i in range(epochs):
            for j in range(len(X_train)):
                self.forward(X_train[j])
                temp_grad = self.back_propogation(X_train[j], y_train[j])
                grad += temp_grad # need to normalize. just a placeholder
            weights = np.concatenate((self.weights_1, self.weights_2, self.weights_3), axis=0)
            weights -= learning_rate * grad
            self.weights_1 = weights[:self.hidden_size_1 * (self.input_size + 1)].reshape(self.hidden_size_1, self.input_size + 1)
            self.weights_2 = weights[self.hidden_size_1 * (self.input_size + 1):self.hidden_size_1 * (self.input_size + 1) + self.hidden_size_2 * (self.hidden_size_1 + 1)].reshape(self.hidden_size_2, self.hidden_size_1 + 1)
            self.weights_3 = weights[self.hidden_size_1 * (self.input_size + 1) + self.hidden_size_2 * (self.hidden_size_1 + 1):].reshape(self.output_size, self.hidden_size_2 + 1)
        pass

    def set_weights(self, weights_1, weights_2, weights_3):
        # for loading weights from file
        self.weights_1 = weights_1
        self.weights_2 = weights_2
        self.weights_3 = weights_3

    def test(self, X_test, y_test, num_classes):
        correct = 0
        total = 0
        for i in range(len(X_test)):
            self.forward(X_test[i])
            prediction = self.predict()
            correct += (prediction == y_test[i])
            total += 1
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.4f}%")
        return accuracy
    
def store_weights(classifier, neural_net, percentage):
    if classifier == 0: # digits dataset
        filename = f"digitWeights/{int(percentage*100)}.npz"
    else:
        filename = f"faceWeights/{int(percentage*100)}.npz"
    path = os.path.join(os.getenv("neural_net_weights_path"), filename)
    np.savez(path, weights_1 = neural_net.weights_1, weights_2 = neural_net.weights_2, weights_2 = neural_net.weights_2)