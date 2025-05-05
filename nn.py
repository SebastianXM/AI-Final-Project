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
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = num_classes

        self.input_layer = np.zeros(self.input_size+1).astype(np.longdouble)
        self.hidden_layer_1 = np.zeros(self.hidden_size_1+1).astype(np.longdouble)
        self.hidden_layer_2 = np.zeros(self.hidden_size_2+1).astype(np.longdouble)
        self.output_layer = np.zeros(self.output_size).astype(np.longdouble)

        self.input_layer[0] = 1
        self.hidden_layer_1[0] = 1
        self.hidden_layer_2[0] = 1

        fan_in_1 = self.input_size + 1
        limit_1 = np.sqrt(1.0 / fan_in_1)
        self.weights_1 = np.random.randn(self.hidden_size_1, fan_in_1).astype(np.longdouble) * limit_1 

        fan_in_2 = self.hidden_size_1 + 1
        limit_2 = np.sqrt(1.0 / fan_in_2)
        self.weights_2 = np.random.randn(self.hidden_size_2, fan_in_2).astype(np.longdouble) * limit_2

        fan_in_3 = self.hidden_size_2 + 1
        limit_3 = np.sqrt(1.0 / fan_in_3)
        self.weights_3 = np.random.randn(self.output_size, fan_in_3).astype(np.longdouble) * limit_3

    def forward(self, x):

        self.input_layer[1:] = x
        self.hidden_layer_1[1:] = sigmoid(np.matmul(self.weights_1, self.input_layer))
        self.hidden_layer_2[1:] = sigmoid(np.matmul(self.weights_2, self.hidden_layer_1))
        self.output_layer = sigmoid(np.matmul(self.weights_3, self.hidden_layer_2))

    def predict(self):
        if len(self.output_layer) == 1:
            return 1 if self.output_layer[0] >= 0.5 else 0
        else:
            return np.argmax(self.output_layer)

    def back_propogation(self, x, y):
        if self.output_size != 1:
            y_hat = np.zeros(self.output_size).astype(np.longdouble)
            y_hat[y] = 1
        else:
            y_hat = y
        delta_4 = self.output_layer - y_hat
        delta_3 = np.matmul(np.transpose(self.weights_3), delta_4)[1:] * sigmoid_derivative(self.hidden_layer_2[1:])
        delta_2 = np.matmul(np.transpose(self.weights_2), delta_3)[1:] * sigmoid_derivative(self.hidden_layer_1[1:])
        grad_1 = np.outer(delta_2, self.input_layer)
        grad_2 = np.outer(delta_3, self.hidden_layer_1)
        grad_3 = np.outer(delta_4, self.hidden_layer_2)
        return grad_1, grad_2, grad_3

    def train(self, X_train, y_train, epochs, learning_rate, regularization):
        X_train = np.reshape(X_train, (len(X_train), -1))
        loss_history = []
        start_time = time.time()
        for i in range(epochs):
            grad_1 = np.zeros_like(self.weights_1)
            grad_2 = np.zeros_like(self.weights_2)
            grad_3 = np.zeros_like(self.weights_3)
            for j in range(len(X_train)):
                self.forward(X_train[j])
                temp_grad1, temp_grad_2, temp_grad3 = self.back_propogation(X_train[j], y_train[j])
                grad_1 += temp_grad1
                grad_2 += temp_grad_2
                grad_3 += temp_grad3
            
            grad_1 /= len(X_train)
            grad_1[:,1:] += (regularization * self.weights_1[:,1:])/ len(X_train)
            grad_2 /= len(X_train)
            grad_2[:,1:] += (regularization * self.weights_2[:,1:])/ len(X_train)
            grad_3 /= len(X_train)
            grad_3[:,1:] += (regularization * self.weights_3[:,1:])/ len(X_train)

            if np.linalg.norm(grad_1) < 0.0001 and np.linalg.norm(grad_2) < 0.0001 and np.linalg.norm(grad_3) < 0.0001:
                print("Converged")
                break

            self.weights_1 -= learning_rate * grad_1
            self.weights_2 -= learning_rate * grad_2
            self.weights_3 -= learning_rate * grad_3
            loss = self.compute_cost(X_train, y_train, regularization)
            loss_history.append(loss)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total Training Time: {total_time:.2f} seconds")
        return total_time
    
    def set_weights(self, weights_1, weights_2, weights_3):
        self.weights_1 = weights_1
        self.weights_2 = weights_2
        self.weights_3 = weights_3

    def test(self, X_test, y_test):
        X_test = np.reshape(X_test, (len(X_test), -1))
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
    
    def compute_cost(self, X, Y, regularization):
        m = X.shape[0]
        total_loss = 0.0
        for i in range(m):
            self.forward(X[i])
            aL = self.output_layer
            y_vec = (np.arange(self.output_size)==Y[i]).astype(float) \
                    if self.output_size>1 else Y[i]
            total_loss += -np.sum(y_vec*np.log(aL) + (1-y_vec)*np.log(1-aL))
        total_loss /= m
        
        reg = (regularization/(2*m)) * (
            np.sum(self.weights_1[:,1:]**2)
        + np.sum(self.weights_2[:,1:]**2)
        + np.sum(self.weights_3[:,1:]**2)
        )
        return total_loss + reg

def rel_error(A, B):
    return np.max(np.abs(A-B) / (np.maximum(1e-8, np.abs(A)+np.abs(B))))
    
def store_weights(classifier, neural_net, percentage, run):
    if classifier == 0: # digits dataset
        filename = f"digitWeights/{int(percentage*100)}/run{run}.npz"
    else:
        filename = f"faceWeights/{int(percentage*100)}/run{run}.npz"
    path = os.path.join(os.getenv("neural_net_weights_path"), filename)
    np.savez(path, weights_1 = neural_net.weights_1, weights_2 = neural_net.weights_2, weights_3 = neural_net.weights_3)

def store_time_or_accuracy(name, time_or_accuracy_list):
    path = os.path.join(os.getenv("neural_net_time_accuracy_path"), f"{name}.npz")
    np.save(path, time_or_accuracy_list)

if __name__ == "__main__":
    digits_X_train, digits_y_train, digits_X_test, digits_y_test, face_X_train, face_y_train, face_X_test, face_y_test = get_data()
    digits_X_train = np.array(digits_X_train).astype(np.longdouble)
    digits_y_train = np.array(digits_y_train)
    face_X_train = np.array(face_X_train).astype(np.longdouble)
    face_y_train = np.array(face_y_train)
    digits_X_test = np.array(digits_X_test).astype(np.longdouble)
    digits_y_test = np.array(digits_y_test)
    face_X_test = np.array(face_X_test).astype(np.longdouble)
    face_y_test = np.array(face_y_test)


    np.divide(digits_X_train, np.amax(digits_X_train), out=digits_X_train)
    np.divide(face_X_train, np.amax(face_X_train), out=face_X_train)

    digits_accuracy_runs = []
    digits_training_time_runs = []
    face_accuracy_runs = []
    face_training_time_runs = []

    num_runs = 5
    percentages = [0.1 * i for i in range(1, 11)]
    for percentage in percentages:
        print(f"Percentage of training data: {int(percentage * 100)}%")
        digits_accs = []
        digits_times = []
        face_accs = []
        face_times = []
        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}")
            num_digits_samples = int(percentage * len(digits_X_train))
            indices = np.random.permutation(len(digits_X_train))
            digits_X_train_subset = digits_X_train[indices[:num_digits_samples]]
            digits_y_train_subset = digits_y_train[indices[:num_digits_samples]]
            num_face_samples = int(percentage * len(face_X_train))
            indices = np.random.permutation(len(face_X_train))
            face_X_train_subset = face_X_train[indices[:num_face_samples]]
            face_y_train_subset = face_y_train[indices[:num_face_samples]]

            digits_nn = NeuralNetwork(28*28, 20, 20, 10)
            face_nn = NeuralNetwork(60*70, 20, 20, 1)
            
            digits_total_time = digits_nn.train(digits_X_train_subset, digits_y_train_subset, 1000, 0.5, 0)
            face_total_time = face_nn.train(face_X_train_subset, face_y_train_subset, 100, 0.7, 0)

            store_weights(0, digits_nn, percentage, run)
            store_weights(1, face_nn, percentage, run)

            digits_test_accuracy = digits_nn.test(digits_X_test, digits_y_test)
            face_test_accuracy = face_nn.test(face_X_test, face_y_test)

            digits_accs.append(digits_test_accuracy)
            digits_times.append(digits_total_time)
            face_accs.append(face_test_accuracy)
            face_times.append(face_total_time)

        digits_accuracy_runs.append(digits_accs)
        digits_training_time_runs.append(digits_times)
        face_accuracy_runs.append(face_accs)
        face_training_time_runs.append(face_times)
        print("========================================")

    
    store_time_or_accuracy("digits_accuracy", digits_accuracy_runs)
    store_time_or_accuracy("digits_training_time", digits_training_time_runs)
    store_time_or_accuracy("face_accuracy", face_accuracy_runs)
    store_time_or_accuracy("face_training_time", face_training_time_runs)

    digits_training_time_means = [np.mean(times) for times in digits_training_time_runs]
    face_training_time_means = [np.mean(times) for times in face_training_time_runs]

    plt.figure(figsize=(8, 6))
    plt.plot([p * 100 for p in percentages], digits_training_time_means, marker='o', label="Digits Average Training Time")
    plt.xlabel("Percentage of Training Data Used (%)")
    plt.ylabel("Average Training Time (seconds)")
    plt.title("Digits Average Training Time vs Percentage of Training Data Used")
    plt.legend()

    plt.figure(figsize=(8, 6))
    plt.plot([p * 100 for p in percentages], face_training_time_means, marker='o', label="Face Average Training Time")
    plt.xlabel("Percentage of Training Data Used (%)")
    plt.ylabel("Average Training Time (seconds)")
    plt.title("Face Average Training Time vs Percentage of Training Data Used")
    plt.legend()

    digits_accuracy_means = [np.mean(accs) for accs in digits_accuracy_runs]
    digits_accuracy_stds = [np.std(accs) for accs in digits_accuracy_runs]
    face_accuracy_means = [np.mean(accs) for accs in face_accuracy_runs]
    face_accuracy_stds = [np.std(accs) for accs in face_accuracy_runs]

    plt.figure(figsize=(8, 6))
    plt.errorbar([p * 100 for p in percentages], digits_accuracy_means, yerr=digits_accuracy_stds, fmt='-o', capsize=5, label="Digits Accuracy (mean ± std)")
    plt.xlabel("Percentage of Training Data Used (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Digits Accuracy (Average ± Std) vs Percentage of Training Data Used")
    plt.legend()

    plt.figure(figsize=(8, 6))
    plt.errorbar([p * 100 for p in percentages], face_accuracy_means, yerr=face_accuracy_stds, fmt='-o', capsize=5, label="Face Accuracy (mean ± std)")
    plt.xlabel("Percentage of Training Data Used (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Face Accuracy (Average ± Std) vs Percentage of Training Data Used")
    plt.legend()

    plt.tight_layout()
    plt.show()