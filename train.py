
from denoising.data import CAEDataset
from denoising.models import ConvAutoencoder
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = transforms.ToTensor()

# Amend following code for my data
train_data = CAEDataset()
test_data = CAEDataset(region_dir='../a_mdt_data/HR_model_data/qtrland_testing_regions')

# train_data = torchvision.datasets.MNIST(
#     root='./data/',
#     train=True,
#     download=True,
#     transform=transforms.ToTensor()
# )
# test_data = torchvision.datasets.MNIST(
#     root='./data/',
#     train=False,
#     download=True,
#     transform=transforms.ToTensor()
# )


num_workers = 0
# batch_size = 64
batch_size = 128

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

model = ConvAutoencoder()
print(model)
print(len(train_loader))
print(len(test_loader))
model.to(device)
criterion = nn.MSELoss(reduction='none')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_epochs = 300

for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    for data in train_loader:
        images = data[0].to(device)
        targets = data[1].to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        land_mask = targets != 0
        loss = loss * land_mask
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*images.size(0)
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))

torch.save(model.state_dict(), 'models/300epochs_qtrland_model_autoencoder.pth')

images, targets = next(iter(test_loader))
images = images.to(device)
output = model(images)
images = images.cpu().numpy()
output = output.view(batch_size, 1, 128, 128)
output = output.detach().cpu().numpy()
output = (output - output.min()) / (output.max() - output.min())
output = output.clip(0, 1)
land_mask = np.array(targets != 0)
output = output * land_mask

fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(25,4))
for images, row in zip([images, output, targets], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='turbo')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()
plt.close()

