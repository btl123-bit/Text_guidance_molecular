import numpy as np
from featurizer import OneHotFeaturizer
from models import MolecularVAE
from rdkit import Chem
import torch.nn.functional as F
from tqdm import tqdm
from scoring_functions import get_scoring_function
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import create_model_and_diffusion
import torch
import functools
from torch.utils.data import Dataset, DataLoader
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from torch.optim import AdamW
import pandas as pd
from transformers import AutoModel
from transformers import AutoTokenizer

class SmilesDataset(Dataset):
    def __init__(self, file_path):

        df = pd.read_csv(file_path, sep='\t')
        self.smiles_list = df['SMILES'].tolist()
        self.context_list = df['description'].tolist()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.smiles_list[idx], self.context_list[idx]


def filter_valid_smiles(batch_tensor, device,VAE_model):
    valid_indices = []
    index = 0
    oh = OneHotFeaturizer()
    sample = VAE_model.channel_decoder(batch_tensor)
    recon_x = VAE_model.decode(sample)
    recon_x = recon_x.cpu().detach().numpy()
    y = np.argmax(recon_x, axis=2)
    for i in range(y.shape[0]):
        updata_smiles = oh.decode_smiles_from_index(y[i])
        mol = Chem.MolFromSmiles(updata_smiles)
        if mol is not None:
            valid_indices.append(index)
        index = index + 1

    filtered_tensor = batch_tensor[valid_indices]
    return filtered_tensor, valid_indices


def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD




def creat_dataset(batch_size,smi_file_path):
    oh = OneHotFeaturizer()
    bert_model = AutoModel.from_pretrained('scibert')
    bert_tokz = AutoTokenizer.from_pretrained('scibert')
    bert_model = bert_model.to("cuda")
    bert_model.eval()

    smiles_dataset = SmilesDataset(smi_file_path)
    with torch.no_grad():
        for i in range(len(smiles_dataset.smiles_list)):
            smiles = smiles_dataset.smiles_list[i]
            smiles = smiles.ljust(120)
            smiles = torch.from_numpy(oh.featurize([smiles]).astype(np.float32)).to('cuda')
            smiles = smiles.squeeze(0)
            smiles_dataset.smiles_list[i] = smiles

            text = smiles_dataset.context_list[i]
            tok_op = bert_tokz(
                text, max_length=216, truncation=True, padding='max_length'
            )
            toked_text = torch.tensor(tok_op['input_ids']).unsqueeze(0)
            toked_text_attentionmask = torch.tensor(tok_op['attention_mask']).unsqueeze(0)
            assert (toked_text.shape[1] == 216)
            lh = bert_model(toked_text.cuda()).last_hidden_state
            smiles_dataset.context_list[i] = {'states':lh,'mask':toked_text_attentionmask.cuda()}   #.to('cpu')


    smiles_dataloader = DataLoader(smiles_dataset.smiles_list, batch_size=batch_size, shuffle=False)
    text_dataloader = DataLoader(smiles_dataset.context_list, batch_size=batch_size, shuffle=False)
    return smiles_dataloader, text_dataloader


def trainer(batch_size,smi_file_path,vae_model_path,device):

    VAE_model = MolecularVAE()
    VAE_model.load_state_dict(torch.load(vae_model_path))
    VAE_model.to(device)
    for param in VAE_model.parameters():  #
        param.requires_grad = False  #

    model, diffusion = create_model_and_diffusion()
    model.to(device)
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
    mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=False, fp16_scale_growth=0.001)
    opt = AdamW(mp_trainer.master_params, lr=0.0001, weight_decay=0.0)

    print("creat dataset ,please wait patiently")

    smiles_dataloader,context_dataloader = creat_dataset(batch_size,smi_file_path)

    for step in range(0, 30001):
        loss_total = 0
        total = 0
        for smiles,context in tqdm(zip(smiles_dataloader, context_dataloader), desc="Training progress"):
            total = total + 1
            start_vec = smiles.transpose(1, 2).to(device)
            context["states"] = context["states"].squeeze(1)
            context["mask"] = context["mask"].squeeze(1)
            mu, logvar = VAE_model.encode(start_vec)
            std = torch.exp(0.5 * logvar)


            eps = 3e-2 * torch.randn_like(std)
            w = eps.mul(std).add_(mu)

            w = VAE_model.channel_encoder(w)
            w, index = filter_valid_smiles(w, device,VAE_model)
            z = w
            context['states'] = context['states'][index]
            context['mask'] = context['mask'][index]

            t, weights = schedule_sampler.sample(z.shape[0], device)
            compute_losses = functools.partial(diffusion.training_losses, model, z, t,context,model_kwargs={})
            losses = compute_losses()
            loss = (losses["loss"] * weights).mean()
            loss.requires_grad_(True)
            loss_total = loss_total + loss.item()

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        print("step: ", step+1)
        print("loss: ",loss_total/total)

        result_loss = loss_total/total
        if step % 500 == 0:
            torch.save(model.state_dict(),
                       f"dm_ckpt/unet_model_{step}_{result_loss}.pth")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer(batch_size = 100,smi_file_path="dataset/train.smi",
            vae_model_path ='vae_ckpt/second_vae-1325-6.995438575744629_1.pth', device = device)
