import time
import random
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data_processing import get_data

def nn_pytorch(X_train, y_train, X_test, y_test, input_size, num_classes, percentage):
    # change to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # three-layer neural network
    class NN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Sigmoid(),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
                nn.Linear(hidden_size, output_size),
            )

        def forward(self, x):
            return self.net(x)
        
    lr = 0.1
    model = NN(input_size, 100, num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # training
    epochs = 10
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X_train)):
            optimizer.zero_grad()
            output = model(X_train[i].view(1, -1))
            loss = criterion(output, y_train[i].view(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X_train)}")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")

    # testing
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(X_test)):
            output = model(X_test[i].view(1, -1))
            _, predicted = torch.max(output.data, 1)
            total += 1
            correct += (predicted == y_test[i]).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.4f}%")
    
    return accuracy, total_time

def pytorch_method():
    digits_X_train, digits_y_train, digits_X_test, digits_y_test, face_X_train, face_y_train, face_X_test, face_y_test = get_data()
    digits_X_train = np.array(digits_X_train)
    digits_y_train = np.array(digits_y_train)
    face_X_train = np.array(face_X_train)
    face_y_train = np.array(face_y_train)

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

            print("Digits Dataset")
            digits_test_accuracy, digits_total_time = nn_pytorch(digits_X_train_subset, digits_y_train_subset, digits_X_test, digits_y_test, 28*28, 10, percentage)
            print("\nFace Dataset")
            face_test_accuracy, face_total_time = nn_pytorch(face_X_train_subset, face_y_train_subset, face_X_test, face_y_test, 60*70, 2, percentage)

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
    pytorch_method()