import IPython
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import functools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import utils
import os

n_filters = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StandardClassifier(nn.Module):
    def __init__(self, n_outputs=1):
        super(StandardClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1 * n_filters, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(1 * n_filters)
        
        self.conv2 = nn.Conv2d(in_channels=1 * n_filters, out_channels=2 * n_filters, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(2 * n_filters)

        self.conv3 = nn.Conv2d(in_channels=2 * n_filters, out_channels=4 * n_filters, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * n_filters)

        self.conv4 = nn.Conv2d(in_channels=4 * n_filters, out_channels=6 * n_filters, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(6 * n_filters)

        self.fc1 = nn.Linear(6 * n_filters * 4 * 4, 512)  # Adjust the size (4x4 here) based on input dimensions
        self.fc2 = nn.Linear(512, n_outputs)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


standard_classifier = StandardClassifier()

#torch.hub.download_url_to_file('https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1', 'train_face.h5')
loader = utils.lab2.TrainingDatasetLoader('train_face.h5')

number_of_training_examples = loader.get_train_size()
(images, labels) = loader.get_batch(100)

face_images  = images[np.where(labels==1)[0]]
not_face_images = images[np.where(labels==0)[0]]

idx_face = 17
idx_not_face = 11

plt.figure(figsize=(5,5))
plt.subplot(1, 2, 1)
plt.imshow(face_images[idx_face])
plt.title('Face'); plt.grid(False)

plt.subplot(1, 2, 2)
plt.imshow(not_face_images[idx_not_face])
plt.title('Not Face'); plt.grid(False)

#plt.show()

params = dict(
    batch_size = 32,
    num_epochs = 2,
    learning_rate = 5e-4,
)

standard_classifier.to(device)

optimizer = optim.Adam(standard_classifier.parameters(), lr=params["learning_rate"])
criterion = torch.nn.BCEWithLogitsLoss() #equivalent of sigmoid cross entropy with logits

step = 0
loss_history = []
for epoch in range(params['num_epochs']):
    epoch_loss = 0
    for idx in tqdm(range(loader.get_train_size()//params["batch_size"])):
        x, y = loader.get_batch(params['batch_size'])
        
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        
        x = x.permute(0, 3, 1 ,2)
        
        logits = standard_classifier(x)
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        epoch_loss += loss.item()
        
batch_x, batch_y = loader.get_batch(5000)

batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)

batch_x = batch_x.permute(0, 3, 1, 2)

standard_classifier.eval()

with torch.no_grad():
    logits = standard_classifier(batch_x)
    predictions = torch.sigmoid(logits)
    y_pred_standard = torch.round(predictions)
    acc_standard = (batch_y == y_pred_standard).float().mean()
    
print(f"Standard CNN accuracy on (potentially biased) training set: {acc_standard.item():.4f}")

test_faces = utils.lab2.get_test_faces()
keys = ["Light Female", "Light Male", "Dark Female", "Dark Male"]
for group, key in zip(test_faces,keys):
  plt.figure(figsize=(5,5))
  plt.imshow(np.hstack(group))
  plt.title(key, fontsize=15)
    
standard_classifier.eval()

# List to store the predicted probabilities for each group
standard_classifier_probs = []

# Loop over each group in test_faces
for group in test_faces:
    # Convert each image in the group to a PyTorch tensor and move to the correct device
    group_tensor = torch.tensor(group, dtype=torch.float32).to(device)

    group_tensor = group_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW

    with torch.no_grad():
        logits = standard_classifier(group_tensor)
        probs = torch.sigmoid(logits)

        # Calculate the mean probability per group and store it
        mean_prob = probs.mean().item()  # Get the mean probability for this group
        standard_classifier_probs.append(mean_prob)

# Plot the prediction accuracies per demographic group
xx = range(len(keys))
yy = standard_classifier_probs
plt.bar(xx, yy)
plt.xticks(xx, keys)
plt.ylim(max(0, min(yy) - (max(yy) - min(yy)) / 2.), max(yy) + (max(yy) - min(yy)) / 2.)
plt.title("Standard classifier predictions")
plt.show()



"===================================================VAE==============================================="




def vae_loss_function(x, x_recon, mu, logsigma, kl_weight=0.0005):
    latent_loss = 1/2 * torch.mean(logsigma.exp() + mu.pow(2) - 1 - logsigma)
    
    reconstruction_loss = torch.mean(torch.abs(x - x_recon))
    
    vae_loss = kl_weight * latent_loss + reconstruction_loss
    
    return vae_loss


def sampling(z_mean, z_logsigma):
    batch, latent_dim = z_mean.shape
    
    epsilon = torch.randn(batch, latent_dim).to(z_mean.device)
    
    z = z_mean + torch.exp(1/2 * z_logsigma) * epsilon
    
    return z

def debiasing_loss_function(x, x_pred, y, y_logit, mu, logsigma):
    
    vae_loss = vae_loss_function(x, x_pred, mu, logsigma)
    
    classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_logit, y)
    
    face_indicator = (y == 1).float()
    
    total_loss = classification_loss + torch.mean(face_indicator)* vae_loss
    
    return total_loss, classification_loss.mean()


n_filters = 12 
latent_dim = 100 


class FaceDecoder(nn.Module):
    def __init__(self, latent_dim=100, n_filters=12):
        super(FaceDecoder, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(latent_dim, 4 * 4 * 6 * n_filters)  # Fully connected layer to reshape to feature map

        self.deconv1 = nn.ConvTranspose2d(6 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(2 * n_filters, 1 * n_filters, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(1 * n_filters, 3, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = x.view(-1, 6 * n_filters, 4, 4)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))

        return x


latent_dim = 100
n_filters = 12
batch_size = 32

decoder = FaceDecoder(latent_dim, n_filters)

z = torch.randn(batch_size, latent_dim)

reconstructed_image = decoder(z)

print(reconstructed_image.shape) 


class DB_VAE(nn.Module):
    def __init__(self, latent_dim):
        super(DB_VAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = StandardClassifier(2 * self.latent_dim + 1)
        
        self.decoder = FaceDecoder()
        
    def encode(self, x):
        encoder_output = self.encoder(x)

        y_logit = encoder_output[:, :1]  

        z_mean = encoder_output[:, 1:self.latent_dim+1]  
        z_logsigma = encoder_output[:, self.latent_dim+1:]  

        return y_logit, z_mean, z_logsigma
    
    def reparameterize(self, z_mean, z_logsigma):
        z = sampling(z_mean, z_logsigma)
        return z
    
    def decode(self, z):
        reconstruction = self.decoder(z)
        return reconstruction
    
    def forward(self, x):
        y_logit, z_mean, z_logsigma = self.encode(x)
        
        z = self.reparameterize(z_mean, z_logsigma)
        
        recon = self.decode(z)
        
        return y_logit, z_mean, z_logsigma, recon
    
    def predict(self, x):
        y_logit, _, _ = self.encode(x)
        
        return y_logit

dbvae = DB_VAE(latent_dim).to(device)

def get_latent_mu(images, dbvae, batch_size=32):
    N = len(images)  
    mu = torch.zeros(N, dbvae.latent_dim).to(device)  
    for start_ind in range(0, N, batch_size):
        end_ind = min(start_ind + batch_size, N)
        batch = images[start_ind:end_ind]

        batch = torch.tensor(batch, dtype=torch.float32) / 255.0
        batch = batch.permute(0, 3, 1, 2).to(device)

        _, batch_mu, _ = dbvae.encode(batch)

        # Store the latent means
        mu[start_ind:end_ind] = batch_mu

    return mu

def get_training_sample_probabilities(images, dbvae, bins=10, smoothing_fac=0.001):
    
    print("Recompiling the sampling probabilities")
    mu = get_latent_mu(images, dbvae)
    
    training_sample_p = np.zeros(mu.shape[0])
    
    for i in range(latent_dim):
        
        latent_distribution = mu[:,i]
        latent_distribution_cpu = latent_distribution.detach().cpu().numpy()
        
        hist_density, bin_edges = np.histogram(latent_distribution_cpu, density=True, bins=bins)

        bin_edges[0] = -float('inf')
        bin_edges[-1] = float('inf')
        
        bin_idx = np.digitize(latent_distribution_cpu, bin_edges)
        
        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)
        
        p = 1.0 / (hist_smoothed_density[bin_idx-1])
        
        p = p / np.sum(p)
        
        training_sample_p = np.maximum(training_sample_p, p)
        
    training_sample_p /= np.sum(training_sample_p)
    
    return training_sample_p

params_DB_VAE = dict(
    batch_size = 32,
    learning_rate = 5e-4,
    latent_dim = 100,
    num_epochs = 1,
)

optimizer_DB_VAE = optim.Adam(dbvae.parameters(), lr=params_DB_VAE["learning_rate"])


def debiasing_train_step(x, y):
    
    x, y = x.to(device), y.to(device)

    y_logit, z_mean, z_logsigma, x_recon = dbvae(x)

    
    loss, class_loss = debiasing_loss_function(x, x_recon, y, y_logit, z_mean, z_logsigma)

    # TODO backward pass
    # ----------------------------------------------------------------------
    optimizer_DB_VAE.zero_grad()
    loss.backward()
    optimizer_DB_VAE.step()

    #-----------------------------------------------------------------------

    return loss.item()  # Return the loss as a scalar


all_faces = loader.get_all_train_faces()

# The training loop
step = 0
for epoch in range(params_DB_VAE["num_epochs"]):
    #IPython.display.clear_output(wait=True)
    print(f"Starting epoch {epoch+1}/{params_DB_VAE['num_epochs']}")

    # Recompute data sampling probabilities for debiasing
    '''TODO: recompute the sampling probabilities for debiasing'''
    p_faces = get_training_sample_probabilities(all_faces, dbvae, bins=10, smoothing_fac=0.01)

    # Loop over the training batches
    for batch_idx in tqdm(range(loader.get_train_size() // params_DB_VAE["batch_size"])):
        # load a batch of data
        (x_data, y_data) = loader.get_batch(params_DB_VAE["batch_size"], p_pos=p_faces)
        x, y = torch.tensor(x_data).to(device), torch.tensor(y_data).to(device)

        x = x.permute(0, 3, 1, 2)


        # Perform a training step (optimize loss)
        loss = debiasing_train_step(x, y)

        # Plot the progress every 200 steps (adjust this frequency as needed)
        if batch_idx % 500 == 0:

            plt.figure(figsize=(2,1))
            plt.subplot(1, 2, 1)

            idx = np.where(y_data==1)[0][0]
            _, _, _, recon = dbvae(x)

            recon = recon.permute(0,2,3,1).cpu().detach()

            plt.imshow(x_data[idx])
            plt.grid(False)

            plt.subplot(1, 2, 2)
            recon = np.clip(recon, 0, 1)
            plt.imshow(recon[idx])
            plt.grid(False)

        step += 1
        
y_logit, z_mean, z_logsigma, x_recon = dbvae(x)
#print(f"x_recon shape:{x_recon.shape}")

dbvae.eval()

# List to store the predicted probabilities for each group
dbvae_probs = []

# Loop over each group in test_faces
for group in test_faces:
    # Convert each image in the group to a PyTorch tensor and move to the correct device
    group_tensor = torch.tensor(group, dtype=torch.float32).to(device)

    group_tensor = group_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW

    with torch.no_grad():
        dbvae_logits = dbvae(group_tensor)
        probs = torch.sigmoid(logits)

        # Calculate the mean probability per group and store it
        mean_prob = probs.mean().item()  # Get the mean probability for this group
        dbvae_probs.append(mean_prob)


# Plotting the results
xx = np.arange(len(keys))
plt.bar(xx, standard_classifier_probs, width=0.2, label="Standard CNN")
plt.bar(xx + 0.2, dbvae_probs, width=0.2, label="DB-VAE")
plt.xticks(xx, keys)
plt.title("Network predictions on test dataset")
plt.ylabel("Probability")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.show()
