import time
import numpy as np
import matplotlib.pyplot as plt
from data_processing import get_data

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = num_classes

        self.weights_1 = np.random.randn(self.input_size, self.hidden_size_1) * np.sqrt(2. / self.input_size)
        self.weights_2 = np.random.randn(self.hidden_size_1, self.hidden_size_2) * np.sqrt(2. / self.hidden_size_1)
        self.weights_3 = np.random.randn(self.hidden_size_2, self.output_size) * np.sqrt(2. / self.hidden_size_2)

        self.bias_1 = np.zeros((1, self.hidden_size_1))
        self.bias_2 = np.zeros((1, self.hidden_size_2))
        self.bias_3 = np.zeros((1, self.output_size))

        self.Z1 = None
        self.Z2 = None
        self.Z3 = None
        self.A1 = None
        self.A2 = None
        self.A3 = None

    def forward(self, x):
        self.Z1 = np.dot(x, self.weights_1) + self.bias_1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.weights_2) + self.bias_2
        self.A2 = relu(self.Z2)
        self.Z3 = np.dot(self.A2, self.weights_3) + self.bias_3
        self.A3 = self.Z3 - np.max(self.Z3, axis=1, keepdims=True)
        exp_scores = np.exp(self.A3)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def back_propogation(self, probs, x, y, regularization=0.0):
        dscores = probs.copy()
        dscores[range(len(y)), y] -= 1
        dscores /= len(y)

        dW3 = np.dot(self.A2.T, dscores)
        db3 = np.sum(dscores, axis=0, keepdims=True)

        dA2 = np.dot(dscores, self.weights_3.T)
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.weights_2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = np.dot(x.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        dW3 += regularization * self.weights_3
        dW2 += regularization * self.weights_2
        dW1 += regularization * self.weights_1

        return dW1, dW2, dW3, db1, db2, db3
    
    def compute_cost(self, probs, Y, regularization):
        correct_logprobs = -np.log(probs[range(len(Y)), Y] + 1e-10)
        data_loss = np.sum(correct_logprobs) / len(Y)
        reg_loss = (0.5 * regularization) * (
            np.sum(self.weights_1**2) +
            np.sum(self.weights_2**2) +
            np.sum(self.weights_3**2)
        )
        return data_loss + reg_loss
        
    def train(self, X_train, y_train, epochs, learning_rate, regularization):
        X_train = X_train.reshape(len(X_train), -1)
        start_time = time.time()
        for i in range(epochs):
            probs = self.forward(X_train)
            loss = self.compute_cost(probs, y_train, regularization)

            dW1, dW2, dW3, db1, db2, db3 = self.back_propogation(probs, X_train, y_train, regularization)

            self.weights_1 -= learning_rate * dW1
            self.weights_2 -= learning_rate * dW2
            self.weights_3 -= learning_rate * dW3
            self.bias_1 -= learning_rate * db1
            self.bias_2 -= learning_rate * db2
            self.bias_3 -= learning_rate * db3

            # print(f"Epoch {i+1}/{epochs}, Loss: {loss:.4f}")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total Training Time: {total_time:.2f} seconds")
        return total_time

    def test(self, X_test, y_test):
        X_test = X_test.reshape(len(X_test), -1)
        probs = self.forward(X_test)
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y_test) * 100
        print(f"Test Accuracy: {accuracy:.4f}%")
        return accuracy

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

            digits_nn = NeuralNetwork(28*28, 100, 100, 10)
            face_nn = NeuralNetwork(60*70, 100, 100, 2)
            
            digits_total_time = digits_nn.train(digits_X_train_subset, digits_y_train_subset, 100, 0.1, 0.0)
            face_total_time = face_nn.train(face_X_train_subset, face_y_train_subset, 100, 0.1, 0)

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

    digits_training_time_means = [np.mean(times) for times in digits_training_time_runs]
    face_training_time_means = [np.mean(times) for times in face_training_time_runs]
    for i in range(len(digits_training_time_means)):
        print(f"Percentage: {percentages[i] * 100:.0f}%, Digits Average Training Time: {digits_training_time_means[i]:.2f} seconds, Face Average Training Time: {face_training_time_means[i]:.2f} seconds")

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
    for i in range(len(digits_accuracy_means)):
        print(f"Percentage: {percentages[i] * 100:.0f}%, Digits Average Accuracy: {digits_accuracy_means[i]:.2f} ± {digits_accuracy_stds[i]:.2f}, Face Average Accuracy: {face_accuracy_means[i]:.2f} ± {face_accuracy_stds[i]:.2f}")

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