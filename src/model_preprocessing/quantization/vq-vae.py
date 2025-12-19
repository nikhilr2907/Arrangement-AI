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




class VQ_VAE_TRAINER(nn.Module):

    def __init__(self, vq_vae_model):
        super().__init__()
        self.vq_vae = vq_vae_model

    def forward(self, x):
        recon_x, z_e, emb = self.vq_vae(x)
        loss = self.vq_vae.loss_function(x, recon_x, z_e, emb)
        return recon_x, loss

    def model_train_step(self, x, optimizer):
        self.train()
        optimizer.zero_grad()
        recon_x, loss = self.forward(x)
        loss.backward()
        optimizer.step()
        return recon_x, loss
    def model_eval_step(self, x):
        self.eval()
        with torch.no_grad():
            recon_x, loss = self.forward(x)
        return recon_x, loss
    def extract_vectors_categories(self):
        """
        Extract the learned codebook embeddings.

        Returns:
            torch.Tensor: Codebook embeddings of shape (emb_dim, num_embeddings)
        """
        self.eval()
        return self.vq_vae.emb.weight.detach().clone()

    def get_encoded_indices(self, x):
        """
        Get the quantized indices for input data.

        Args:
            x: Input tensor

        Returns:
            indices: Tensor of discrete indices for each encoded position
        """
        self.eval()
        with torch.no_grad():
            z_e = self.vq_vae.encode(x)
            _, indices = self.vq_vae.emb(z_e, weight_sg=True)
        return indices
    def save_model_weights(self, save_path: str):
        """
        Save the model weights to a file.

        Args:
            save_path: Path to save the model weights
        """
        torch.save(self.vq_vae.state_dict(), save_path)
        print(f"Saved model weights to {save_path}")
    def save_embeddings(self, save_path: str):
        """
        Save the learned codebook embeddings to a file.

        Args:
            save_path: Path to save the embeddings
        """
        embeddings = self.extract_vectors_categories()
        torch.save({
            'embeddings': embeddings,
            'emb_size': self.vq_vae.emb_size,
            'hidden': self.vq_vae.hidden,
        }, save_path)
        print(f"Saved embeddings of shape {embeddings.shape} to {save_path}")

    def load_embeddings(self, load_path: str):
        """
        Load previously saved embeddings into the model.

        Args:
            load_path: Path to load the embeddings from
        """
        checkpoint = torch.load(load_path)
        self.vq_vae.emb.weight.data = checkpoint['embeddings']
        print(f"Loaded embeddings of shape {checkpoint['embeddings'].shape} from {load_path}")
    
    def full_training_sequence(self,train_loader,optimizer,num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, data in enumerate(train_loader):
                inputs = data
                recon_x, loss = self.model_train_step(inputs, optimizer)
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        

