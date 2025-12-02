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

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flt = nn.Linear(36 * 5 * 5, 900)
        
        self.flc1 = nn.Linear(900, 128)
        self.flc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = x.view(-1, 36 * 5 * 5)
        
        x = self.flt(x)
        x = F.relu(x)
        
        x = self.flc1(x)
        x = F.relu(x)
        
        x = self.flc2(x)
        
        
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
cnn_model = SimpleCNN()
cnn_model = cnn_model.to(device)

train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transform, download=True)

train_images = train_dataset.data.numpy()
train_labels = train_dataset.targets.numpy()

#from torchinfo import summary
#model_summary = summary(cnn_model, input_size=(1, 1, 28, 28), verbose=0)
#print(model_summary)

optimizer = optim.Adam(cnn_model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
accuracy_metric = Accuracy(task='multiclass', num_classes=10).to(device)

BATCH_SIZE = 64
EPOCHS = 5

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

loss_history = utils.util.LossHistory(smoothing_factor=0.95)
plotter = utils.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss', scale='semilogy')

for epoch in range(EPOCHS):
    
    cnn_model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    
    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = cnn_model(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        loss_history.append(loss.item())
        plotter.plot(loss_history.get())
        
        running_loss += loss.item()
        
        accuracy = accuracy_metric(outputs, labels)
        running_accuracy += accuracy.item()
    avg_loss = running_loss / len(train_loader)
    avg_accuracy = running_accuracy / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    
cnn_model.eval()

correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = cnn_model(inputs)
        
        _, predictions = torch.max(outputs, dim=1)
        
        correct_predictions += (predictions == labels).sum().item()
        
        total_samples += labels.size(0)

test_acc = correct_predictions / total_samples
print('Test Accuracy: ', test_acc)    

test_images = test_dataset.data.numpy()
test_labels = test_dataset.targets.numpy()

cnn_model.eval()
predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        
        device_inputs, device_labels = inputs.to(device), labels.to(device)
        
        outputs = cnn_model(device_inputs)
        predictions += outputs.cpu().numpy().tolist()    

print(predictions[0])
print("Label of this digit is:", test_labels[0])
plt.imshow(test_images[0,:,:], cmap=plt.cm.binary)
plt.show()

#@title Change the slider to look at the model's predictions! { run: "auto" }

image_index = 18 #@param {type:"slider", min:0, max:100, step:1}
plt.subplot(1,2,1)
utils.lab2.plot_image_prediction(image_index, predictions, test_labels, test_images)
plt.subplot(1,2,2)
utils.lab2.plot_value_prediction(image_index, predictions,  test_labels)

# Plots the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red

num_rows = 5
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  utils.lab2.plot_image_prediction(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  utils.lab2.plot_value_prediction(i, predictions, test_labels)

plt.show()