import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions = None):
    """
    plots training data, test data and compares the predictions
    """
    plt.figure(figsize=(10,7))
    
    #plot data in blue
    
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    
    #plot test in green
    
    plt.scatter(test_data, test_labels, c="g", s=4, label = "Testing Data")
    
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label = "Predictions")
        
    plt.legend(prop={"size": 14});
    
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.weights * x + self.bias #linear regression formula

device = "cuda" if torch.cuda.is_available() else "cpu"

#creating known parameters
weight = 0.7
bias = 0.3

#create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

X[:10], y[:10]
#print(len(X), len(y))

#train / test split

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

#print(len(X_train), len(y_train), len(X_test), len(X_test))

#===================================== Predicting test data ==================================

plot_predictions(X_train, y_train, X_test, y_test);
#plt.show()

#===================================== Testing nn ==================================
torch.manual_seed(42)
model_0 = LinearRegressionModel()
#print(list(model_0.parameters()))
#print(model_0.state_dict())
#model_0.to(device)
#print(next(model_0.parameters()).device)

y_preds = model_0(X_test)
#print(y_preds)

with torch.inference_mode():
    y_preds = model_0(X_test)
    
print(y_preds)
print(y_test)
plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds)
#plt.show()

#===================================== Training nn ==================================

#loss function
loss_fn = nn.L1Loss()

#optimizer (stochastic gradient descent)

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)

torch.manual_seed(42)

epochs = 200

#X_train = X_train.to(device)
#y_train = y_train.to(device)
#X_test = X_test.to(device)
#y_test = y_test.to(device)

#tracking values

epoch_count = []
loss_values = []
test_loss_values = []

###Training
for epoch in range(epochs):
    
    model_0.train()
    
    #Forward Pass
    
    y_pred = model_0(X_train)
    
    #Loss calculation
    
    loss = loss_fn(y_pred, y_train)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        
        test_loss = loss_fn(test_pred, y_test)
    
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        
np.array(torch.tensor(loss_values).numpy()), torch.tensor(test_loss_values).numpy()
    
plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();

with torch.inference_mode():
    y_preds_new = model_0(X_test)
    
plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds);
plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds_new);
plt.show()
    
