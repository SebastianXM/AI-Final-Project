import time
import random
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from data_processing import get_data

load_dotenv()

def perceptron_train(classifier, X_train, y_train, input_size, num_classes, percentage):

    weights = []
    for i in range(num_classes):
        weights.append(np.zeros(input_size+1))
    X_train = np.reshape(X_train, (len(X_train), -1))
    max_epochs = 100
    no_change = False
    start_time = time.time()
    for i in range(max_epochs):
        if no_change:
            break
        no_change = True
        for j in range(len(X_train)):
            augmented_x = np.insert(X_train[j], 0, 1)
            if num_classes == 1:
                prediction = np.dot(weights[0], augmented_x)
                if prediction >= 0 and y_train[j] == 0:
                    weights[0] -= augmented_x
                    no_change = False
                elif prediction < 0 and y_train[j] == 1:
                    weights[0] += augmented_x
                    no_change = False
                else:
                    continue
            else:
                prediction = (-1, -np.inf)
                for k in range(num_classes):
                    temp_prediction = np.dot(weights[k], augmented_x)
                    if temp_prediction > prediction[1]:
                        prediction = (k, temp_prediction)
                
                if prediction[0] != y_train[j]:
                    weights[prediction[0]] -= augmented_x
                    weights[y_train[j]] += augmented_x
                    no_change = False
    end_time = time.time()
    total_time = end_time - start_time
    filename = ""
    if classifier == 0: # digits dataset
        filename = f"digitWeights/{int(percentage*100)}.npy"
    else:
        filename = f"faceWeights/{int(percentage*100)}.npy"
    path = os.path.join(os.getenv("perceptron_weights_path"), filename)
    np.save(path, weights)
    return total_time
    

def perceptron_test(classifier, X_test, y_test, num_classes, percentage):
    correct = 0
    total = len(X_test)
    weights = []
    filename = ""
    if classifier == 0: # digits dataset
        filename = f"digitWeights/{int(percentage*100)}.npy"
    else:
        filename = f"faceWeights/{int(percentage*100)}.npy"
    path = os.path.join(os.getenv("perceptron_weights_path"), filename)
    weights = np.load(path, allow_pickle=True)
    X_test = np.reshape(X_test, (len(X_test), -1))

    for i in range(len(X_test)):
        augmented_x = np.insert(X_test[i], 0, 1)
        if num_classes == 1:
            prediction = np.dot(weights[0], augmented_x)
            if (prediction >= 0 and y_test[i] == 1) or (prediction < 0 and y_test[i] == 0):
                correct += 1
        else:
            prediction = (-1, -np.inf)
            for k in range(num_classes):
                temp_prediction = np.dot(weights[k], augmented_x)
                if temp_prediction > prediction[1]:
                    prediction = (k, temp_prediction)
            if prediction[0] == y_test[i]:
                correct += 1

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.4f}%")
    return accuracy

def perceptron_method():
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

            digits_total_time = perceptron_train(0, digits_X_train_subset, digits_y_train_subset, 28*28, 10, percentage)
            face_total_time = perceptron_train(1, face_X_train_subset, face_y_train_subset, 60*70, 1, percentage)

            digits_test_accuracy = perceptron_test(0, digits_X_test, digits_y_test, 10, percentage)
            face_test_accuracy = perceptron_test(1, face_X_test, face_y_test, 1, percentage)

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

if __name__ == "__main__":
    perceptron_method()