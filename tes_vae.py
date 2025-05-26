import numpy as np
from featurizer import OneHotFeaturizer
from models import MolecularVAE
from polygon.utils.utils import canonicalize_list
import torch
from scoring_functions import get_scoring_function


if __name__ == '__main__':

    oh = OneHotFeaturizer()
    MolecularVAE = MolecularVAE()
    MolecularVAE.load_state_dict(torch.load('vae_ckpt/second_vae-1325-6.995438575744629_1.pth'))
    MolecularVAE.to('cuda')
    for param in MolecularVAE.parameters():  #
        param.requires_grad = False  #
    subset = ["C1=CC=C2C(=C1)C(=O)C(=CO2)/C=N/NC(=O)C3=CN=CC=C3"]

    smile_list = []
    for j in range(200):
        for i in range(len(subset)):
            #print(i)
            smiles_test = subset[i]
            start_vec = torch.from_numpy(oh.featurize([smiles_test]).astype(np.float32)).to('cuda')
            start_vec = start_vec.transpose(1, 2).to("cuda")
            mu, logvar = MolecularVAE.encode(start_vec)
            #w = mu

            std = torch.exp(0.5 * logvar)  # 计算标准差
            eps = 3e-2 * torch.randn_like(std)  # 生成与标准差相同形状的随机噪声
            w = eps.mul(std).add_(mu)  # 生成一个正太分布的采样，

            z = MolecularVAE.channel_encoder(w)
            w = MolecularVAE.channel_decoder(z)

            recon_x = MolecularVAE.decode(w)
            recon_x = recon_x.cpu().detach().numpy()
            y = np.argmax(recon_x, axis=2)

            end = oh.decode_smiles_from_index(y[0])

            smile_list.append(end)

    canonicalized_smiles = list(canonicalize_list(smile_list, include_stereocenters=True))
    scoring_function_kwargs = {}

    scoring_function = get_scoring_function(scoring_function="tanimoto", num_processes=0, **scoring_function_kwargs)
    scores = scoring_function(canonicalized_smiles)
    print(len(canonicalized_smiles))
    # for i in range(len(canonicalized_smiles)):
    #     print(canonicalized_smiles[i])

    for score in scores:
        # 如果分数为0，则表示生成的分子是不合理的，只保留合理的分数 以及 对应 的编码
        if score != 0:
            print("smiles:{}   score:{}".format(canonicalized_smiles[i], scores[i]))
        i = i + 1
