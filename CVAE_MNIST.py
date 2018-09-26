# Simple script to generate MNIST images with a conditional variational autoencoder
# Template: https://github.com/pytorch/examples/tree/master/vae

# Import modules
import imageio
import os
import torch
import torch.utils.data
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Parameters
batchSize = 32
conditionalSize = 10
epochs = 50
hiddenSize = 400
latentSize = 20

# Run on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loaders
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batchSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batchSize, shuffle=True)


# Neural network architecture
class CVAE(nn.Module):
    def __init__(self, conditional_size, hidden_size, latent_size):
        super(CVAE, self).__init__()

        # Network layers
        self.fc1 = nn.Linear(28 * 28 + conditional_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size + conditional_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 28 * 28)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h1 = self.relu(self.fc1(inputs))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h3 = self.relu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 28*28), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


# Instantiate neural network
net = CVAE(conditionalSize, hiddenSize, latentSize).to(device)

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):

    # Binary cross entropy
    ble = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784),
                                             reduction='sum')

    # KL divergence
    # Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return ble + kld


# Convert numbers to one-hot vectors
def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets


# Training
def network_training(n):
    net.train()
    train_loss = 0
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = one_hot(labels, 10).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = net(data, labels)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if i % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                n, i * len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          n, train_loss / len(train_loader.dataset)))


# Testing
def network_testing(n):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = one_hot(labels, 10).to(device)
            recon_batch, mu, logvar = net(data, labels)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                m = min(data.size(0), 8)
                comparison = torch.cat([data[:m],
                                        recon_batch.view(batchSize, 1, 28, 28)[:m]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(n) + '.png', nrow=m)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# Optimize neural network
for epoch in range(1, epochs+1):
    network_training(epoch)
    network_testing(epoch)
    with torch.no_grad():
        c_test = torch.eye(10, 10).to(device)
        z_test = torch.randn(10, 20).to(device)
        samples = net.decode(z_test, c_test)
        save_image(samples.view(10, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png', nrow=10)

# Generate gif file
gif = []
for filename in ['results/sample_'+str(i)+'.png' for i in range(1, epochs+1)]:
    gif.append(imageio.imread(filename))
imageio.mimsave('results/CVAE_MNIST.gif', gif[-20:])
for file in [x for x in os.listdir('results') if x[-3:] == 'png']:
    os.remove('results/' + file)
