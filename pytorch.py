import time
import torch
import torch.nn as nn
import numpy as np
from dotenv import load_dotenv
import os
from data_processing import get_data
import matplotlib.pyplot as plt

load_dotenv()

class PyTorchNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.num_classes),
        )

    def forward(self, x):
        return self.net(x)

    def train_model(self, X_train, y_train, epochs=10, lr=0.1):
        X_train_flat = np.reshape(X_train, (X_train.shape[0], -1))
        X_train_tensor = torch.tensor(X_train_flat, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.train()
        start_time = time.time()

        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, X_train_tensor.size(0)):
                inputs = X_train_tensor[i].view(1, -1)
                labels = y_train_tensor[i].view(1)

                optimizer.zero_grad()     
                outputs = self(inputs)     
                loss = criterion(outputs, labels)
                loss.backward()            
                optimizer.step()          

                total_loss += loss.item()

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total Training Time: {total_time:.2f} seconds")
        return total_time

    def test_model(self, X_test, y_test):

        X_test_flat = np.reshape(X_test, (X_test.shape[0], -1))
        X_test_tensor = torch.from_numpy(X_test_flat).float()
        y_test_tensor = torch.from_numpy(y_test).long()  

        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(len(X_test_tensor)):
                inputs = X_test_tensor[i].view(1, -1)
                labels = y_test_tensor[i] 
                outputs = self(inputs)   
                _, predicted = torch.max(outputs.data, 1)
                total += 1
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.4f}%")
        return accuracy
    
def store_pytorch_model(classifier, model, percentage, run):
    filename = ""
    if classifier == 0: # digits dataset
        filename = f"digitWeights/{int(percentage*100)}/run{run}.pth"
    else:
        filename = f"faceWeights/{int(percentage*100)}/run{run}.pth"
    path = os.path.join(os.getenv("pytorch_weights_path"), filename)
    torch.save(model.state_dict(), path)

def store_time_or_accuracy(name, time_or_accuracy_list):
    path = os.path.join(os.getenv("pytorch_time_accuracy_path"), f"{name}.npz")
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

            digits_nn = PyTorchNN(28*28, 100, 10)
            digits_total_time = digits_nn.train_model(digits_X_train_subset, digits_y_train_subset, epochs=10, lr=0.1)
            store_pytorch_model(0, digits_nn, percentage, run)
            digits_test_accuracy = digits_nn.test_model(digits_X_test, digits_y_test)
            
            face_nn = PyTorchNN(60*70, 100, 2)
            face_total_time = face_nn.train_model(face_X_train_subset, face_y_train_subset, epochs=10, lr=0.1)
            store_pytorch_model(1, face_nn, percentage, run)
            face_test_accuracy = face_nn.test_model(face_X_test, face_y_test)

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