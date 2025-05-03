import time
import random
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from data_processing import get_data

load_dotenv()


class Perceptron:
    def __init__(self, input_size, num_classes):
        weights = []
        for i in range(num_classes):
            weights.append(np.zeros(input_size+1)) # one perceptron for each output class
        self.weights = weights
        self.input_size = input_size
        self.num_classes = num_classes

    def set_weights(self, weights):
        self.weights = weights

    def predict(self, augmented_x):
        prediction = -1
        prediction_value = -np.inf
        if self.num_classes == 1:
            prediction_value = np.dot(self.weights[0], augmented_x)
            if prediction_value >= 0:
                prediction = 1
            else:
                prediction = 0
        else:
            for k in range(self.num_classes):
                temp_prediction = np.dot(self.weights[k], augmented_x)
                if temp_prediction > prediction_value:
                    prediction_value = temp_prediction
                    prediction = k
        return prediction
    
    def update_weights(self, augmented_x, prediction, y_train):
        if self.num_classes == 1:
            if prediction != y_train:
                if y_train == 1:
                    self.weights[0] += augmented_x
                else:
                    self.weights[0] -= augmented_x
        else:
            if prediction != y_train:
                self.weights[prediction] -= augmented_x
                self.weights[y_train] += augmented_x

    def train(self, X_train, y_train, max_epochs=100):
        X_train = np.reshape(X_train, (len(X_train), -1))
        no_change = False 
        start_time = time.time()
        for i in range(max_epochs):
            if no_change:
                break
            no_change = True
            for j in range(len(X_train)):
                augmented_x = np.insert(X_train[j], 0, 1)
                prediction = self.predict(augmented_x)
                if prediction != y_train[j]:
                    self.update_weights(augmented_x, prediction, y_train[j])
                    no_change = False
        end_time = time.time()
        total_time = end_time - start_time
        return total_time
    
    def test(self, X_test, y_test):
        correct = 0
        total = len(X_test)
        X_test = np.reshape(X_test, (len(X_test), -1))
        for i in range(len(X_test)):
            augmented_x = np.insert(X_test[i], 0, 1)
            prediction = self.predict(augmented_x)
            if prediction == y_test[i]:
                correct += 1
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.4f}%")
        return accuracy
    
def store_weights(classifier, perceptron, percentage, run):
    filename = ""
    if classifier == 0: # digits dataset
        filename = f"digitWeights/{int(percentage*100)}/run{run}.npz"
    else:
        filename = f"faceWeights/{int(percentage*100)}/run{run}.npz"
    path = os.path.join(os.getenv("perceptron_weights_path"), filename)
    np.savez(path, weights=np.array(perceptron.weights)) 

def store_time_or_accuracy(name, time_or_accuracy_list):
    path = os.path.join(os.getenv("perceptron_time_accuracy_path"), f"{name}.npz")
    np.save(path, time_or_accuracy_list)

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

            digits_perceptron = Perceptron(28*28, 10)
            face_perceptron = Perceptron(60*70, 1)
            
            digits_total_time = digits_perceptron.train(digits_X_train_subset, digits_y_train_subset, 100)
            face_total_time = face_perceptron.train(face_X_train_subset, face_y_train_subset, 100)

            store_weights(0, digits_perceptron, percentage, run)
            store_weights(1, face_perceptron, percentage, run)

            digits_test_accuracy = digits_perceptron.test(digits_X_test, digits_y_test)
            face_test_accuracy = face_perceptron.test(face_X_test, face_y_test)

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