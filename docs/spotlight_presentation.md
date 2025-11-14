---
marp: true
theme: gaia
paginate: true
size: 16:9
style: |
  h1 { color:rgb(201, 97, 101); }
  h2 { color: rgb(204, 106, 57); margin:0; }
  h3 { color: rgb(177, 136, 47); margin:0;}
  ul, ol { margin:0; padding-left: 1.5em; }
  li { margin:0; }
---

<!-- _class: lead invert -->
# A Contrastive Approach to Weight Space Learning

DJ Swanevelder

Supervised by: Ruan van der Merwe (Bytefuse)

---

<style>
.columns {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}
</style>

## Context

<div class="columns">

<div>

### Contrastive Learning
![width:500px](latex/fig/contrastive.png)

</div>

<div>

### Weight Space Learning
- 1 point $\rightarrow$ 1 model
- Discriminative
  - Predict model properties
- Generative
  - High-performing Model Weights
</div>

</div>

---



## Problem

<style scoped>
.slide-container {
  display: flex;
  justify-content: center;
  align-items: center;
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  margin-top:0px; /* Adjust this value to move image up/down */
}
</style>

<div class="slide-container">

![width:1100px](latex/fig/pipeline.png)

</div>

---
## Conditional Model Sampling

<style scoped>
.slide-container {
  display: flex;
  justify-content: center;
  align-items: center;
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  margin-top:0px; /* Adjust this value to move image up/down */
}
</style>

<div class="slide-container">

![width:1100px](latex/fig/conditional_model_sampling.png)

</div>

---

## PCA + Weight Autoencoder

<style scoped>
.slide-fig {
  display: flex;
  justify-content: center;
  align-items: center;
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  margin-top:75px; /* Adjust this value to move image up/down */
}
</style>

<div class="slide-fig">

![width:1200px](latex/fig/pca_ae.png)

</div>



---
## Results
### Weight Autoencoder 
<style scoped>
.two-img-row {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  gap: 40px;
  margin-bottom: 10px;
}
.two-img-row > div {
  text-align: center;
}
</style>

<div class="two-img-row">
  <div>
    <img src="latex/fig/weight_encoder/combined_loss_plot.png" alt="Loss" width="485px"><br>
    <span style="font-size: 16px;">(a) Loss</span>
  </div>
  <div>
    <img src="latex/fig/weight_encoder/output_comparison.png" alt="Random Input Output Comparison" width="450px"><br>
    <span style="font-size: 16px;">(b) Output Comparison (random input)</span>
  </div>
</div>

---

## Results
### Weight Autoencoder
<style scoped>
.centered-table-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  position: absolute;
  top: 90px;
  left: 0;
  right: 0;
  bottom: 0;
}
.table-label {
  font-size: 30px;
  font-weight: bold;
  margin-bottom: 10px;
  text-align: center;
}
</style>

<div class="centered-table-container">
  <div class="table-label">Output comparison (real data)</div>
  <table style="margin: 0 auto; width: 200px; font-size: 10px;">
    <thead>
      <tr>
        <th>Metric</th>
        <th>Mean Value</th>
        <th>Range</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Cosine Similarity</td>
        <td>-0.082</td>
        <td>[−0.547, 0.526]</td>
      </tr>
      <tr>
        <td>Correlation</td>
        <td>0.016</td>
        <td>[−0.329, 0.344]</td>
      </tr>
      <tr>
        <td>Prediction Agreement</td>
        <td>31.21%</td>
        <td>[2.06%, 64.88%]</td>
      </tr>
    </tbody>
  </table>
</div>

---
## Results
### Shared Encoder

<style scoped>
.two-img-row {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  gap: 40px;
  margin-bottom: 10px;
}
.two-img-row > div {
  text-align: center;
}
</style>

<div class="two-img-row">
  <div>
    <img src="latex/fig/shared/nt_xent_loss_plot.png" alt="Random Input Output Comparison" width="550px"><br>
    <span style="font-size: 16px;">(a) Loss</span>
  </div>
  <div>
    <img src="latex/fig/money_shot.png" alt="Loss" width="485px"><br>
    <span style="font-size: 16px;">(b) Input vs Output Accuracy</span>
  </div>
</div>

---
## Conclusion
<div class="columns">

<div>

### Contribution
- First joint modeling of $P(W | D, R)$
- Demonstrates viability of contrastive alignment for heterogeneous modalities


</div>

<div>

### Future Work 
- Improve weight encoder
  - Reverse PCA
  - PCA + Non-linear Stages
- Sequential Autoencoder for Neural Embeddings
- Dataset distillation 
- Variational Autoencoder 
</div>

</div>

---
<!-- _class: lead invert -->
# Thank you

---

![width:800px](latex/fig/weight_encoder/hidden_dim_loss.png)