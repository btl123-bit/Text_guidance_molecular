#利用训练好的VAE模型和扩散模型，采样分子
import torch as th
import numpy as np
from featurizer import OneHotFeaturizer
from models import MolecularVAE
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from affinity.affinity_score import pred_affinity
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
import re
from polygon.utils.utils import canonicalize_list
import csv
import torch
from transformers import AutoModel
from transformers import AutoTokenizer


class OptResult:
    def __init__(self, smiles: str, score: float) -> None:
        self.smiles = smiles
        self.score = score
    def __eq__(self, other):
        return (self.score, self.smiles) == (other.score, other.smiles)
    def __lt__(self, other):
        return (self.score, self.smiles) < (other.score, other.smiles)

def validate_smiles(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return 0
        return 1
    except:
        return 0

def sample_smiles(batch_size,desc_text):
    sample_fn = diffusion.p_sample_loop
    smiles_list = []
    text = desc_text

    #加载文本编码模型
    bert_model = AutoModel.from_pretrained('scibert')
    bert_tokz = AutoTokenizer.from_pretrained('scibert')
    bert_model = bert_model.to("cuda")
    bert_model.eval()

    tok_op = bert_tokz(
        text, max_length=216, truncation=True, padding='max_length'
    )
    toked_text = torch.tensor(tok_op['input_ids']).unsqueeze(0)
    toked_text_attentionmask = torch.tensor(tok_op['attention_mask']).unsqueeze(0)
    assert (toked_text.shape[1] == 216)
    lh = bert_model(toked_text.cuda()).last_hidden_state

    toked_text_attentionmask = toked_text_attentionmask.repeat(batch_size,1,1)
    lh = lh.repeat(batch_size,1,1)

    #context = {'states': lh, 'mask': toked_text_attentionmask.cuda()}
    context = {'states': lh, 'mask': None}


    sample,list = sample_fn(
            model,
            (batch_size, 3, 16, 16),
            context,
            clip_denoised=True,
            model_kwargs={},
            device="cuda",
            progress=True
        )
    chunk_tensors = torch.chunk(sample, 10)
    for sample in chunk_tensors:
        sample = prior_model.channel_decoder(sample)
        recon_x = prior_model.decode(sample)
        recon_x = recon_x.cpu().detach().numpy()
        y = np.argmax(recon_x, axis=2)
        updata_smiles = oh.decode_smiles_from_index(y[0])
        smiles_list.append(updata_smiles)

    smiles_list = list(canonicalize_list(smiles_list, include_stereocenters=True))
    print(len(smiles_list))

    return smiles_list



if __name__ == "__main__":
    oh = OneHotFeaturizer()
    prior_model = MolecularVAE()
    prior_model.load_state_dict(torch.load("vae_ckpt/second_vae-1325-6.995438575744629_1.pth"))
    prior_model.to('cuda')
    prior_model.eval()

    model, diffusion = create_model_and_diffusion()
    model.load_state_dict(th.load("dm_ckpt/unet_model_0.0017587485490366817.pth"))
    model.to("cuda")
    model.eval()

    desc_text = "The molecule is a tetracarboxylic acid that is benzene substituted by four carboxy groups at positions 1, 2, 4 and 5 respectively. It is a member of benzoic acids and a tetracarboxylic acid."
    batch_size =  10
    smiles_gen = sample_smiles(batch_size,desc_text)

    for i in range(len(smiles_gen)):
        print(smiles_gen[i])

        #O=C(O)c1cc(O)c(C(=O)O)c(=O)cc1C(=O)O












