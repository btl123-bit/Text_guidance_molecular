
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
from scoring_functions import get_scoring_function
from polygon.utils.utils import canonicalize_list
import torch.nn as nn
import torch.optim as optim

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
    #torch.autograd.set_detect_anomaly(True)
    sample_fn = diffusion.p_sample_loop

    text = desc_text
    bert_model = AutoModel.from_pretrained('scibert')
    bert_tokz = AutoTokenizer.from_pretrained('scibert')
    bert_model = bert_model.to("cuda")
    for param in bert_model.parameters():
        param.requires_grad = False

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

    for epoch in range(1000):
        print("epoch: ",epoch)
        smiles_list = []

        sample,list = sample_fn(
                model,
                (batch_size, 3, 16, 16),
                context,
                clip_denoised=True,
                model_kwargs={},
                device="cuda",
                progress=True
            )
        #//////////////////////////////////////////////////////////////////////////////////////////////
        sample.detach()
        chunk_tensors = torch.chunk(sample, batch_size)
        for sample in chunk_tensors:
            sample = prior_model.channel_decoder(sample)
            recon = prior_model.decode(sample)
            recon_x = recon.cpu().detach().numpy()
            y = np.argmax(recon_x, axis=2)
            updata_smiles = oh.decode_smiles_from_index(y[0])
            smiles_list.append(updata_smiles)
        #affinity
        #gen_scores = pred_affinity(smiles_list, "hybrid/data/raw/gen_smi.csv")
        # gen_scores = gen_scores.tolist()

        # simity
        scores = scoring_function(smiles_list)

        i =0
        selected_indices = []
        for score in scores:
            if score != 0:
                print("smiles:{}   score:{}".format(smiles_list[i] , scores[i]))
                selected_indices.append(i)
            i = i+1
        scores = scores[selected_indices]
        print("len: ", len(scores))
        if len(scores) == 0:
            continue
        # //////////////////////////////////////////////////////////////////////////////////////////////


        log_selected_prob_sum = torch.zeros(len(selected_indices), 120, 1).to("cuda")

        log_prob_list = []
        for sample_x in list:
            sample_x = sample_x[selected_indices]
            sample_x = prior_model.channel_decoder(sample_x)
            recon = prior_model.decode(sample_x)

            prob = recon
            recon = torch.log(prob + 1e-8)
            log_prob_list.append(recon)

        actions = log_prob_list[9].argmax(dim=-1)

        for log_prob in log_prob_list:
            log_selected_prob = log_prob.gather(dim=-1, index=actions.unsqueeze(-1))
            log_selected_prob_sum += log_selected_prob

        loss = -(1/10)*(log_selected_prob_sum / 10 - (1/Variable(scores)) ).mean()
        print(loss)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()



if __name__ == "__main__":
    oh = OneHotFeaturizer()
    prior_model = MolecularVAE()
    prior_model.load_state_dict(torch.load("vae_ckpt/second_vae-1325-6.995438575744629_1.pth"))
    prior_model.to('cuda')
    for param in prior_model.parameters():
        param.requires_grad = False

    model, diffusion = create_model_and_diffusion()
    model.load_state_dict(th.load("dm_ckpt/unet_model_0.0017587485490366817.pth"))
    model.to("cuda")
    for param in model.parameters():  #
        param.requires_grad = True  #

    scoring_function_kwargs = {}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scoring_function = get_scoring_function(scoring_function="tanimoto", num_processes=0, **scoring_function_kwargs)

    desc_text = "The molecule is a tetracarboxylic acid that is benzene substituted by four carboxy groups at positions 1, 2, 4 and 5 respectively. It is a member of benzoic acids and a tetracarboxylic acid."
    batch_size =  10
    smiles_gen = sample_smiles(batch_size,desc_text)













