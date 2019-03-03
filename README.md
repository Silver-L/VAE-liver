# beta-VAE for liver
A implementation of beta-VAE and VAE(beta = 1) for liver

## Requirements
```
1. TensorFlow >= 1.4.0
2. SimpleITK
3. tqdm
4. matplotlib
5. scipy
```

## Usage
Input: TFRecord file\
Training Model: trainer.py\
Reconstructing Image: predict_gen.py\
Generating Image: predict_spe.py

## Reference
[1] Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes, (Ml), 1–14. https://doi.org/10.1051/0004-6361/201527329

[2] Thiagarajan, B. G., Member, A., & Voyiadjis, G. Z. (2016). Β-Vae: Learning Basic Visual Concepts With a Constrained Variational Framework. Iclr 2017, (July), 1–13.

[3] https://github.com/wuga214/IMPLEMENTATION_Variational-Auto-Encoder
