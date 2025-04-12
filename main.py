import torch 
import torch.nn as nn

from data_processing import loadDataFile, loadLabelsFile

# load digits data
digits_train_data = loadDataFile("data/digitdata/trainingimages", 5000, 28, 28)
digits_train_labels = loadLabelsFile("data/digitdata/traininglabels", 5000)
digits_test_data = loadDataFile("data/digitdata/testimages", 1000, 28, 28)
digits_test_labels = loadLabelsFile("data/digitdata/testlabels", 1000)
digits_val_data = loadDataFile("data/digitdata/validationimages", 1000, 28, 28)
digits_val_labels = loadLabelsFile("data/digitdata/validationlabels", 1000)

# load face data
face_train_data = loadDataFile("data/facedata/facedatatrain", 451, 60, 70)
face_train_labels = loadLabelsFile("data/facedata/facedatatrainlabels", 451)
face_test_data = loadDataFile("data/facedata/facedatatest", 150, 60, 70)
face_test_labels = loadLabelsFile("data/facedata/facedatatestlabels", 150)
face_val_data = loadDataFile("data/facedata/facedatavalidation", 301, 60, 70)
face_val_labels = loadLabelsFile("data/facedata/facedatavalidationlabels", 301)

# create features
digits_X_train = [[[0 if pixel == 0 else 1 for pixel in row] for row in data.getPixels()] for data in digits_train_data]
digits_X_test = [[[0 if pixel == 0 else 1 for pixel in row] for row in data.getPixels()] for data in digits_test_data]
digits_X_val = [[[0 if pixel == 0 else 1 for pixel in row] for row in data.getPixels()] for data in digits_val_data]

face_X_train = [[[0 if pixel == 0 else 1 for pixel in row] for row in data.getPixels()] for data in face_train_data]
face_X_test = [[[0 if pixel == 0 else 1 for pixel in row] for row in data.getPixels()] for data in face_test_data]
face_X_val = [[[0 if pixel == 0 else 1 for pixel in row] for row in data.getPixels()] for data in face_val_data]
face_X_train = [sample[:60] for sample in face_X_train]
face_X_test = [sample[:60] for sample in face_X_test]
face_X_val = [sample[:60] for sample in face_X_val]

# create labels
digits_y_train = digits_train_labels
digits_y_test = digits_test_labels
digits_y_val = digits_val_labels

face_y_train = face_train_labels
face_y_test = face_test_labels
face_y_val = face_val_labels

def nn_pytorch(digits_X_train, digits_y_train, digits_X_test, digits_y_test, digits_X_val, digits_y_val,
               face_X_train, face_y_train, face_X_test, face_y_test, face_X_val, face_y_val):
    # change to tensors
    digits_X_train = torch.tensor(digits_X_train, dtype=torch.float32)
    digits_X_test = torch.tensor(digits_X_test, dtype=torch.float32)
    digits_X_val = torch.tensor(digits_X_val, dtype=torch.float32)

    face_X_train = torch.tensor(face_X_train, dtype=torch.float32)
    face_X_test = torch.tensor(face_X_test, dtype=torch.float32)
    face_X_val = torch.tensor(face_X_val, dtype=torch.float32)

    digits_y_train = torch.tensor(digits_y_train, dtype=torch.long)
    digits_y_test = torch.tensor(digits_y_test, dtype=torch.long)
    digits_y_val = torch.tensor(digits_y_val, dtype=torch.long)

    face_y_train = torch.tensor(face_y_train, dtype=torch.long)
    face_y_test = torch.tensor(face_y_test, dtype=torch.long)
    face_y_val = torch.tensor(face_y_val, dtype=torch.long)

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
        
    # digit data 
    num_classes = 10
    lr = 0.001
    momentum = 0.9

    model = NN(28*28, 100, num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    
    # training
    epochs = 10
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(digits_X_train)):
            optimizer.zero_grad()
            output = model(digits_X_train[i].view(1, -1))
            loss = criterion(output, digits_y_train[i].view(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(digits_X_train)}")

    # testing
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(digits_X_test)):
            output = model(digits_X_test[i].view(1, -1))
            _, predicted = torch.max(output.data, 1)
            total += 1
            correct += (predicted == digits_y_test[i]).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.4f}%")

    # face data
    num_classes = 2
    lr = 0.001
    momentum = 0.9

    model = NN(60*70, 100, num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    # training
    epochs = 10
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(face_X_train)):
            optimizer.zero_grad()
            output = model(face_X_train[i].view(1, -1))
            loss = criterion(output, face_y_train[i].view(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(face_X_train)}")

    # testing
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(face_X_test)):
            output = model(face_X_test[i].view(1, -1))
            _, predicted = torch.max(output.data, 1)
            total += 1
            correct += (predicted == face_y_test[i]).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.4f}%")

if __name__ == "__main__":
    nn_pytorch(digits_X_train, digits_y_train, digits_X_test, digits_y_test, digits_X_val, digits_y_val,
               face_X_train, face_y_train, face_X_test, face_y_test, face_X_val, face_y_val)