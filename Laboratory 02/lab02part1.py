import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import utils
from torchmetrics.classification import Accuracy
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transform, download=True)

train_images = train_dataset.data.numpy()
train_labels = train_dataset.targets.numpy()

plt.figure(figsize=(10,10))
random_indcs = np.random.choice(len(train_images),36,replace=False)
for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image_ind = random_indcs[i]
    plt.imshow(np.squeeze(train_images[image_ind]), cmap = plt.cm.binary)
    plt.xlabel(train_labels[image_ind])

#plt.show()

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128,10)
)

#print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()
accuracy_metric = Accuracy(task='multiclass', num_classes=10).to(device)

BATCH_SIZE = 64
EPOCHS = 5

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.to(device)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()
        
        curr_accuracy = accuracy_metric(outputs, labels)
        running_accuracy += curr_accuracy.item()
        
    avg_loss = running_loss / len(train_loader)
    avg_accuracy = running_accuracy / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    
model.eval()
correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        
        _, predictions = torch.max(outputs, dim=1)
        
        correct_predictions += (predictions == labels).sum().item()
        
        total_samples += labels.size(0)
test_acc = correct_predictions / total_samples
print('Test Accuracy: ', test_acc)
        