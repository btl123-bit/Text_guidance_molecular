## Paper Data

The ChEBI-20 dataset used in this project can be found [here](https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data).

[Scibert](https://huggingface.co/allenai/scibert_scivocab_uncased) and put it into the folder `scibert`.

## Training

1. Train the encoder

   ```bash
   python train_vae.py

2. Training diffusion model
   
   Please create a new folder dm_ckpt to save the trained model.
   
    ```bash
   python train_diffusion.py

## Sample

     ```bash
     sec_sample.py

## Optimize 

Please run the RLLL.py file to optimize the generative model based on the generation task.
  

