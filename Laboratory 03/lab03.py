import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torchinfo import summary
import utils
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List
from tqdm.auto import tqdm
from utils.lab3 import train_step, test_step
import sys
sys.stdout.reconfigure(encoding="utf-8")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
image_path = utils.lab3.download_data(destination="pizza_steak_sushi")
#print(image_path)

train_dir = image_path / "train"
test_dir = image_path / "test"

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
automatic_transforms = weights.transforms()

train_dataloader, test_dataloader, class_names = utils.lab3.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=automatic_transforms, 
    batch_size=32
)

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

for param in model.features.parameters():
    param.requires_grad = False
    
set_seeds()

model.classifier = torch.nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features = len(class_names), bias=True).to(device)
)

summary(model,
        input_size=(32, 3, 224, 224), 
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter()

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        writer.add_scalars(main_tag="Loss",
                           tag_scalar_dict={"train_loss": train_loss,
                                            "test_loss": test_loss},
                           global_step=epoch)

        writer.add_scalars(main_tag="Accuracy",
                           tag_scalar_dict={"train_acc": train_acc,
                                            "test_acc": test_acc},
                           global_step=epoch)

        writer.add_graph(model=model,
                         input_to_model=torch.randn(32, 3, 224, 224).to(device))

    
    writer.close()

    

    
    return results

set_seeds()
""" results = train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=5,
                device=device) """

#print(results)

effnetb3_weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
effnetb3 = torchvision.models.efficientnet_b3(weights=effnetb3_weights)
print(f"In features: {len(effnetb3.classifier.state_dict()['1.weight'][0])}")

OUT_FEATURES = len(class_names)

def create_effnetb3():
    weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
    model = torchvision.models.efficientnet_b3(weights=weights).to(device)
    
    for param in model.features.parameters():
        param.requires_grad = False
    
    set_seeds()
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1536, out_features=OUT_FEATURES)  # B3 has 1536
    ).to(device)
    
    model.name = "effnetb3"
    print(f"[INFO] Created new {model.name} model.")
    return model