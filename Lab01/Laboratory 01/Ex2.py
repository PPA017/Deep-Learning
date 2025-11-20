import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    
    def forward(self, x : torch.Tensor)->torch.Tensor:
        return self.weights * x + self.bias



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
    
    

torch.manual_seed(42)

model_1 = LinearRegressionModel()

#print(f"The model's state dict(): {model_1.state_dict()}")
 
#===================================== EX 3 ==================================

weight = 0.3
bias = 0.9

start = 0
end = 1
step = 0.01
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(y))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

y_preds = model_1(X_test)

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

torch.manual_seed(42)

epochs = 300

epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    model_1.train()
    
    y_pred = model_1(X_train)
    
    loss = loss_fn(y_pred, y_train)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    model_1.eval()
    
    with torch.inference_mode():
        test_pred = model_1(X_test)
        
        test_loss = loss_fn(test_pred, y_test)
    
    if epoch % 20 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        #print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")

#===================================== EX 4 ==================================

    
np.array(torch.tensor(loss_values).numpy()), torch.tensor(test_loss_values).numpy()

plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();

model_1.eval()

with torch.inference_mode():
    y_preds_new = model_1(X_test)
    
plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds_new);
#plt.show()

#===================================== EX 5 ==================================

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "exercises_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME     

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH)

loaded_model_1 = LinearRegressionModel()
loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model_1.state_dict()

loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_1(X_test)
    
print(f"Original predictions: {y_preds_new[:10]}")
print(f"Loaded predictions: {loaded_model_preds[:10]}")