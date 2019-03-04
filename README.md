# beta-VAE for liver
A implementation of beta-VAE and VAE(beta = 1) for Level Set Distribtuion Model(LSDM) of liver 

## Requirements
```
1. TensorFlow >= 1.4.0
2. SimpleITK
3. tqdm
4. matplotlib
5. scipy
```
## Results
### The performance of Model was evaluated with generalization and specificity indices.
* We compared our model to the conventional LSDM based on PCA

* Generalization and Specificity

<img src="https://github.com/Silver-L/VAE-liver/blob/master/result/GEN.jpg" width="297" height="289" alt="error"/><img src="https://github.com/Silver-L/VAE-liver/blob/master/result/SPE.jpg" width="297" height="289" alt="error"/>

* Latent Space (bule: training data, orange/pink: test data)
<img src="https://github.com/Silver-L/VAE-liver/blob/master/result/latent_distribution.PNG" width="350" height="263" alt="error"/>

## Usage
Input: TFRecord file\
Training Model: trainer.py\
Evaluate Model: evaluations.py\
Plot Latent Space: plot_latent_space.py\
Reconstructing Image: predict_gen.py\
Generating Image: predict_spe.py

## Reference
[1] Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes, (Ml), 1–14. https://doi.org/10.1051/0004-6361/201527329

[2] Thiagarajan, B. G., Member, A., & Voyiadjis, G. Z. (2016). Β-Vae: Learning Basic Visual Concepts With a Constrained Variational Framework. Iclr 2017, (July), 1–13.

[3] https://github.com/wuga214/IMPLEMENTATION_Variational-Auto-Encoder
