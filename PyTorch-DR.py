import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.optim as optim
from sklearn.model_selection import KFold

"""
Making three layer neural network to classify 
handwritten digits using PyTorch and Adam algorithim. 
    - Optimized by using GPU.
    - Using Z-score standardization for
    fatser convergance.
    - Using MNIST digit dataset.
    - Use mini-batch (64 size of batch) to optimize GPU usage
    - Use cross validation with five folds
"""

# Load data
data = pd.read_csv("/train.csv")
data = np.array(data)
np.random.shuffle(data)

X = data[:, 1:] 
Y = data[:, 0] # Target value column

# Standardize data 
X = ( X - np.mean(X)) / np.std(X)

# Transform into tensors 
X = torch.tensor(X, dtype=torch.float32)
# Use LongTensor because CEL expects labels to be of type 'torch.long'
Y = torch.tensor(Y, dtype=torch.long)

# Processing Mini-batch
dataset = TensorDataset(X, Y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Parameters and hyperparameters:
epochs = 50
alpha = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(784, 24)
        self.l2 = nn.Linear(24, 10)
        self.l3 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x): # PyTorch expects forward name
        # Apply ReLU to every layer 
        output = self.relu(self.l1(x))
        output = self.relu(self.l2(output))
        output = self.relu(self.l3(output))
        return output
    
# Intialize the neural network model and its loss
model = NeuralNet().to(device=device)
criteria = nn.CrossEntropyLoss()

# Use Adam algorithim for our neural network: model.parameters = weights and biases
optimizer = optim.Adam(model.parameters(), lr=alpha)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

# Training model and perform cross validation for accuracy
for fold, (train_index, test_index) in enumerate(kfold.split(X)):

    # Initialize model for each fold
    model = NeuralNet().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=alpha)

    # grab subset in data
    sub_train = Subset(dataset, train_index) 
    sub_test = Subset(dataset, test_index)

    # process into batches
    subtrain_loader = DataLoader(sub_train, 64, shuffle=True)
    subtest_loader = DataLoader(sub_test, 64, shuffle=True)

    # List that holds all scores for the epochs training cycles
    fold_accuracies = []


    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        # enumarate for logging and train on subset of training data
        for images, labels in subtrain_loader: 
            # Flatten images to 784 and to original color channel
            images = images.reshape(-1, 28*28).to(device=device)
            labels = labels.to(device=device)

            # Compute foward-prop
            outputs = model(images)
            loss = criteria(outputs, labels)

            # Back-prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Take it out of train mode for metrics
        model.eval()
        val_correct = 0
        val_samples = 0 
        with torch.no_grad():
            for images, labels in subtest_loader:
                images, labels = images.to(device=device), labels.to(device=device)
                output = model(images)
                # return max inside data among rows (1)
                _, prediction = torch.max(output.data, 1)
                val_correct += (prediction == labels).sum().item()
                val_samples += labels.size(0) 

        # Turn into percentage
        val_scores = (val_correct / val_samples) * 100 

        # append list that keeps track of accuracy scores / percentages for specfic fold
        fold_accuracies.append(val_scores)

        if epoch % 10 == 0:
            print(f"Fold {fold}, Epoch {epoch}, Validation Accuracy: {val_scores:.3f}%")

    # calculate average outside loop
    avg_fold_accuracy = sum(fold_accuracies) / len(fold_accuracies)

    # add average of that specfic fold to cross validation score list
    cv_scores.append(avg_fold_accuracy)

cv_score_avg = sum(cv_scores) / len(cv_scores)
print(f"Cross Validation scores: {[round(scores, 3) for scores in cv_scores]} \n Average of CV Scores: {cv_score_avg:.3f}")

# Set the model to evaluation mode
model.eval() 

# Testing model with training data to check for overfitting
total_correct = 0
total_samples = 0
with torch.no_grad():
    for images, labels in train_loader:
        # Flatten images to 784 and to original color channel
        images = images.reshape(-1, 28*28).to(device=device) 
        labels = labels.to(device=device)

        prediction = model(images)
        # Get index with max value (output_value, index)
        _, predicted = torch.max(prediction, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0) # Add number of labels in batch to total

# Average it out
accuracy = (total_correct / total_samples) * 100
print(f"Number of training images: {total_samples}\nAccuracy of neural net on training data: {accuracy:.4f}")




