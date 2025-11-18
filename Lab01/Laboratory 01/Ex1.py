import torch
from torch import nn
import matplotlib.pyplot as plt

def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions = None):
    plt.figure(figsize=(10,7))
    
    plt.scatter(train_data, train_labels, c = "b", s = 4, label = "Training Data")
    
    plt.scatter(test_data, test_labels, c = "g", s = 4, label = "Testing Data")
    
    if predictions is not None:
        plt.scatter(test_data, predictions, c = "r", s = 4, label = "Predictions")
    
    plt.legend(prop={"size": 14});
    
    

weight = 0.3
bias = 0.9

start = 0
end = 1
step = 0.01
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(f"Number of X samples: {len(X)}")
print(f"Number of y samples: {len(y)}")
print("First 10 examples of X & y: ")
print(f"X: {X[:10]}")
print(f"y: {y[:10]}")

train_split = int(0.8 * len(y))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

plot_predictions(X_train, y_train, X_test, y_test)
plt.show()