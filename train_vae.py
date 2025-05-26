import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
from featurizer import OneHotFeaturizer
from models import MolecularVAE
from torch.utils.data import Dataset, DataLoader

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


class SmilesDataset(Dataset):
    def __init__(self, file_path):

        with open(file_path, 'r') as f:
            self.smiles_list = f.read().splitlines()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        oh = OneHotFeaturizer()
        smiles = self.smiles_list[idx]
        start = smiles.ljust(120)
        start_vec = torch.from_numpy(oh.featurize([start]).astype(np.float32)).to('cuda')
        start_vec = start_vec.squeeze(0)
        return start_vec



def train(epoch):
    model.train()
    train_loss = 0
    total_channel_loss = 0

    for batch_idx, smiles in enumerate(train_loader):
        start_vec = smiles.transpose(1,2).to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar,channel_x = model(start_vec)
        mu_1  = mu.detach()
        channel_loss = criterion(mu_1, channel_x)

        channel_loss_rounded = round(channel_loss.item(), 6)
        loss_1 = loss_function(recon_batch, start_vec.transpose(1, 2), mu, logvar)
        loss = loss_1 + channel_loss

        loss.backward()
        train_loss += loss_1
        total_channel_loss += channel_loss_rounded
        optimizer.step()

    print(epoch)
    print('train', train_loss / len(train_loader.dataset))
    print('channel_train', total_channel_loss / len(train_loader.dataset))
    return train_loss / len(train_loader.dataset)




if __name__ == '__main__':
    smiles_dataset = SmilesDataset("dataset/1111.smi")
    train_loader = DataLoader(smiles_dataset, batch_size=100, shuffle=True)
    torch.manual_seed(40)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MolecularVAE()
    model.load_state_dict(torch.load('vae_ckpt/vae-103-6.896378517150879_successful_channel.pth'))
    model.to('cuda')
    for param in model.parameters():  #
        param.requires_grad = True  #

    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.MSELoss()

    for epoch in range(1, 1000 + 1):
        train_loss = train(epoch)
        if train_loss < 30 :
            torch.save(model.state_dict(),
                       'vae_ckpt/six_vae-{:03d}-{}_1.pth'.format(epoch, train_loss))
