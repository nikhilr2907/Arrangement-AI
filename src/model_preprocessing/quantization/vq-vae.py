import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model_preprocessing.quantization.nearest_embed import NearestEmbed


class VQ_VAE(nn.Module):
    """Vector Quantized AutoEncoder"""

    def __init__(self, hidden=200, k=10, vq_coef=0.2, comit_coef=0.4, input_size=None, batch_size=None):
        super(VQ_VAE, self).__init__()

        self.emb_size = k
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden = hidden
        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.ce_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0

        self.encoder_spatial_shape = None

        self.fc1 = nn.Linear(self.input_size, 400)
        self.fc2 = nn.Linear(3200, hidden)
        self.fc3 = nn.Linear(hidden, 3200)
        self.decoder_fc1 = nn.Linear(100, 400)
        self.fc4 = nn.Linear(400, self.input_size)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.decoder_conv2 = nn.ConvTranspose2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.emb = NearestEmbed(k, self.emb_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def encode(self, x):
        batch_size = x.shape[0]
        h1 = self.relu(self.fc1(x))
        print("shape after fc1 and relu:", h1.shape)
        h2 = self.relu(self.conv1(h1.view(batch_size, 1, 20, 20)))
        print("shape after conv1 and relu:", h2.shape)
        h3 = self.pool1(h2)
        print("shape after pool1:", h3.shape)
        h4 = self.relu(self.conv2(h3))
        print("shape after conv2 and relu:", h4.shape)
        h5 = self.pool2(h4)
        print("shape after pool2:", h5.shape)
        self.encoder_spatial_shape = h5.shape[1:]
        h6 = self.fc2(h5.view(batch_size, -1))
        print("shape after fc2:", h6.shape)
        return h6.view(batch_size, self.emb_size, int(self.hidden / self.emb_size))

    def decode(self, z):
        batch_size = z.shape[0]
        print("shape of z in decode:", z.shape)
        d1 = self.relu(self.fc3(z))
        print("shape after fc3 and relu:", d1.shape)
        d2 = d1.view(batch_size, *self.encoder_spatial_shape)
        print("shape after reshaping d1:", d2.shape)
        d3 = F.interpolate(d2, scale_factor=(1, 2), mode='nearest')
        print("shape after interpolation 1:", d3.shape)
        d4 = self.relu(self.decoder_conv1(d3))
        print("shape after decoder conv1 and relu:", d4.shape)
        d5 = F.interpolate(d4, scale_factor=(1, 2), mode='nearest')
        print("shape after interpolation 2:", d5.shape)
        d6 = self.decoder_conv2(d5)
        print("shape after decoder conv2:", d6.shape)
        d7 = d6.view(batch_size, -1)
        print("shape after flattening d6:", d7.shape)
        d8 = self.relu(self.decoder_fc1(d7))
        print("shape after decoder fc1 and relu:", d8.shape)
        d9 = self.fc4(d8)
        print("shape after final fc4:", d9.shape)
        return self.tanh(d9)

    def forward(self, x):
        z_e = self.encode(x)
        z_q, _ = self.emb(z_e, weight_sg=True)
        print("shape of z_q before decode:", z_q.shape)
        emb, _ = self.emb(z_e.detach())
        z_q = z_q.view(-1, self.hidden)
        
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
        self.ce_loss = F.mse_loss(recon_x, x)
        self.vq_loss = F.mse_loss(emb, z_e.detach())
        self.commit_loss = F.mse_loss(z_e, emb.detach())
        return self.ce_loss + self.vq_coef * self.vq_loss + self.comit_coef * self.commit_loss

    def latest_losses(self):
        return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss, 'commitment': self.commit_loss}



# if __name__ == "__main__":
#     model = VQ_VAE(input_size=500, batch_size=20)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     x = torch.randn(500, 25, 128)
#     x_reshaped = x.view(x.shape[0], -1)
#     model = VQ_VAE(input_size=x_reshaped.shape[1], batch_size=x_reshaped.shape[0])

#     for i in range(5):
#         for start_len in range(10):
#             print(f"Iteration {i}, Batch {start_len}")
#             x_reshaped = x[start_len*50:(start_len+1)*50].view(50, -1)
#             recon_x, z_e, emb = model(x_reshaped)
#             print("Reconstructed x shape:", recon_x.shape)
#             print("Encoded z_e shape:", z_e.shape)
#             print("Embedding shape:", emb.shape)
#             print("Original x shape:", x_reshaped.shape)
#             loss = model.loss_function(x_reshaped, recon_x, z_e, emb)
#             print("Loss:", loss.item())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
