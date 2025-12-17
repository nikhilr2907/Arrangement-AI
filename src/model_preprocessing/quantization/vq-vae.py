import torch 
import torch.nn as nn
import torch.nn.functional as F
from src.model_preprocessing.quantization.nearest_embed import NearestEmbed

class VQ_VAE(nn.Module):
    """Vector Quantized AutoEncoder for mnist"""

    def __init__(self, hidden=200, k=10, vq_coef=0.2, comit_coef=0.4, **kwargs):
        super(VQ_VAE, self).__init__()

        self.emb_size = k
        self.fc1 = nn.Linear(800, 400)
        self.fc2 = nn.Linear(400, hidden)
        self.fc3 = nn.Linear(hidden, 400)
        self.fc4 = nn.Linear(400, 800)

        self.emb = NearestEmbed(k, self.emb_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.hidden = hidden
        self.ce_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0
        # Convolutional layers for 
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4),dilation=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4),dilation=3)
        self.decoder_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4),dilation=3)#
        self.decoder_conv2 = nn.ConvTranspose2d(64, 1, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4),dilation=3)#
        self.pooling = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.conv1(h1.view(-1, 1, 40, 800)))
        h3 = self.pooling(h2)
        h4 = self.relu(self.conv2(h3))
        h5 = self.pooling(h4)
        h6 = self.fc2(h5.view(-1, self.emb_size, int(self.hidden / self.emb_size)))
        return h6

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h3 = self.relu(self.decoder_conv1(h3.view(-1, 128, 7, 7)))
        h3 = self.decoder_conv2(h3)
        h4 = self.sigmoid(self.fc4(h3))
        return self.tanh(self.fc4(h4))

    def forward(self, x):
        z_e = self.encode(x.view(-1,800))
        z_q, _ = self.emb(z_e, weight_sg=True).view(-1, self.hidden)
        emb, _ = self.emb(z_e.detach()).view(-1, self.hidden)
        return self.decode(z_q), z_e, emb

    def sample(self, size):
        sample = torch.randn(size, self.emb_size,
                             int(self.hidden / self.emb_size))
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        sample = self.decode(emb(sample).view(-1, self.hidden)).cpu()
        return sample

    def loss_function(self, x, recon_x, z_e, emb):
        self.ce_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784))
        self.vq_loss = F.mse_loss(emb, z_e.detach())
        self.commit_loss = F.mse_loss(z_e, emb.detach())

        return self.ce_loss + self.vq_coef*self.vq_loss + self.comit_coef*self.commit_loss

    def latest_losses(self):
        return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss, 'commitment': self.commit_loss}



if __name__ == "__main__":
    model = VQ_VAE()
    x = torch.randn(20, 25, 128)
    recon_x, z_e, emb = model(x)
    loss = model.loss_function(x, recon_x, z_e, emb)
    print("Loss:", loss.item())