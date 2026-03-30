# A Contrastive Approach to Weight Space Learning: From Failure to Function-Preserving Embeddings

*A technical walkthrough of embedding neural network weights into a shared latent space -- and what it took to make conditional weight generation actually work.*

---

## The Big Idea

What if you could generate neural network weights on demand? Not by training a model from scratch, but by describing what you want -- a dataset, a target accuracy -- and sampling weights that achieve it.

This is the problem of learning the joint distribution $p(W \mid \mathcal{D}, R, \text{Arch})$: given a dataset $\mathcal{D}$, desired results $R$, and a fixed architecture, produce weights $W$ that satisfy the specification. The approach is to embed datasets, performance metrics, and trained weight vectors into a shared latent space using contrastive alignment, then decode from that space back to functional weights.

If it works, it opens a path toward meta-learning systems that skip the inner training loop entirely.

## V1: The Original Attempt

The first version (part of the MSc report) used ResNet18 models trained on ImageNet subsets. The pipeline was:

1. **Model Zoo**: Train ~1000 ResNet18 models on various ImageNet class subsets.
2. **PCA Compression**: Flatten each model's 11.7M parameters and reduce to ~13,750 dimensions via PCA.
3. **Autoencoder**: Compress PCA vectors further into a latent space.
4. **NT-Xent Contrastive Loss**: Align weight embeddings with (dataset, results) embeddings.
5. **Conditional Sampling**: Given a new dataset and target accuracy, project into latent space and decode to weights.

The result: a mean output cosine similarity of **-0.082** between original and reconstructed model outputs. Conditional sampling produced flat accuracy curves indistinguishable from random chance. The system learned nothing useful.

## What Went Wrong

Three compounding failures:

**PCA destroys functional information.** Neural network weight spaces are highly nonlinear. PCA, a linear method, crushed 11.7M parameters down to 13,750 dimensions -- an **851:1 compression ratio**. The directions that matter for model function are not the directions of maximum variance.

**MSE on raw weights does not preserve model function.** Two weight vectors can be close in L2 distance but produce completely different outputs, and vice versa. The autoencoder was optimising the wrong objective.

**Permutation symmetry was ignored.** A hidden layer with $H$ neurons has $H!$ equivalent weight configurations (any permutation of neurons gives identical function). Without accounting for this, the weight space contains massive redundancy that confuses any embedding method.

Additionally, only 1000 models provided far too little training signal for the contrastive encoder to learn meaningful alignment.

## V2: Nine Improvements

V2 scaled down to validate the core idea before scaling back up. The architecture was simplified to small MLPs (784-16-3) trained on 3-class MNIST subsets, with nine targeted changes:

1. **Git Re-Basin weight alignment** -- Collapse permutation symmetries by solving a linear assignment problem (Hungarian algorithm) to align all hidden neurons to a reference model before encoding. This removes the $16! \approx 2 \times 10^{13}$ equivalent representations per model.

2. **5x more training data** -- 5,000 weight vectors (1,500 tasks x 3 snapshots each) instead of 1,000.

3. **Longer AE training with early stopping** -- 400 epochs maximum with patience of 60, versus a fixed short schedule.

4. **VAE instead of deterministic AE** -- A variational autoencoder with low KL weight ($\beta = 0.0001$) produces a smooth, interpolable latent space rather than scattered point embeddings.

5. **Richer conditioning** -- 7-dimensional metrics vector (train loss, test loss, test accuracy, log learning rate, optimizer type one-hot, epoch fraction) instead of 3D.

6. **Larger batch size + temperature annealing** -- Batch size 128 with temperature decaying from 0.5 to 0.05 over training, plus soft contrastive labels based on actual similarity.

7. **Retrieval-augmented generation** -- At inference time, project the target condition to latent space, find the K=5 nearest zoo models by cosine similarity, decode all candidates, and return the best.

8. **Post-hoc fine-tuning** -- Apply 3 SGD steps on target-task training data to the decoded weights, correcting small errors in the generation.

9. **24:1 compression ratio** -- 12,611 weight parameters compressed to 512 latent dimensions, versus the original 851:1.

## Results

The improvements compound dramatically:

| Metric | V1 (ResNet/ImageNet) | V2 Final (9 improvements) |
|---|---|---|
| Output Cosine Similarity | -0.082 | **0.895** |
| Mean Accuracy Error | N/A (random) | **0.040** |
| Conditional Sampling R^2 (direct decode) | ~0 | **0.532** |
| Conditional Sampling R^2 (fine-tuned) | ~0 | **0.709** |

### Autoencoder Reconstruction Quality

The VAE preserves model function through compression. Reconstructed models achieve nearly identical accuracy to originals, with a mean output cosine similarity of 0.895 across 100 test models.

![Weight AE reconstruction quality -- original vs. reconstructed accuracy (left) and output cosine similarity distribution (right)](../data/v2/results/reconstruction_quality.png)

### Conditional Model Sampling

The key result: given a target accuracy, the system generates weights that track the target. Three strategies are evaluated -- direct decoding, retrieval-augmented generation, and fine-tuning. Fine-tuning after direct decode is the strongest, with generated models closely following the y=x ideal line for target accuracies above 0.6.

![Conditional sampling -- desired vs. actual accuracy (left) and loss (right) across three generation strategies](../data/v2/results/conditional_sampling.png)

### Training Dynamics

The contrastive training loss tells the story of the iterative improvements. V1 (dashed grey) stalled at high loss. Early V2 runs (green, magenta) overfit severely. The final configuration (cyan) converges to low validation loss while maintaining a negative generalisation gap -- it is actually generalising.

![Training loss comparison across all runs (left) and generalisation gap over time (right)](../data/v2/results/progress_comparison.png)

## What Made the Biggest Difference

Not all nine improvements contributed equally. In rough order of impact:

1. **Compression ratio** -- Going from 851:1 to 24:1 was probably the single largest factor. You cannot reconstruct what you have thrown away.
2. **Git Re-Basin alignment** -- Removing permutation symmetry made the weight space learnable. Without it, the autoencoder has to implicitly learn all $H!$ equivalent representations.
3. **VAE with functional loss** -- A smooth latent space plus a training signal that penalises functional divergence (not just weight MSE) ensures the decoder produces working models.
4. **Richer conditioning** -- Telling the system *how* a model was trained (optimizer, learning rate, epoch) in addition to *what* it achieved made the contrastive alignment substantially more informative.

## What's Next

This validates the core framework on a toy problem. The path forward:

- **Wider accuracy range** -- The current zoo clusters around high accuracy. A broader distribution would stress-test conditional generation at low-accuracy targets, where the current system is weakest.
- **Harder datasets** -- FashionMNIST and CIFAR-10 subsets with larger MLPs or small CNNs.
- **Scaling to larger architectures** -- The ultimate goal remains ResNet-scale models. This likely requires hierarchical compression or weight-space diffusion models rather than a monolithic VAE.
- **Conditional diffusion in latent space** -- Replace direct decoding with a denoising diffusion process conditioned on (dataset, results), which should produce higher-fidelity samples from the learned latent space.

---

*Code and experiments: [msc_mlai](https://github.com/djswanevelder/msc_mlai)*
