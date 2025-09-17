---
theme: gaia
paginate: true
style: |
  .columns {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 2em;
  }
  h1 {
    font-size: 2.5em;
    color: #da6362;
  }
  section.lead h1 {
    text-align: center;
  }
  h2 {
    color: #d77f48;
  }
  h3 {
    color: #bf983d;
    margin-bottom: 0;
    padding-bottom: 0;
  }
  p {
    margin-bottom: 0;
    padding-bottom: 0;
  }
  h3 + ul {
    margin-top: 0;
  }
---
<!-- _class: invert lead -->
# A Contrastive Approach to Weight Space Learning
DJ Swanevelder
Supervised by : Ruan van der Merwe (*Bytefuse*)

---

## Context

<div class="columns">
  <div>
    <h3>Contrastive Learning</h3>
    <ul>
    <li>Meaningful Representations
    <li>Similiar vs Dissimliar</li>
        <ul>
        <li>NLP → Semantic Relationship (Word2Vec)</li>
        </ul>
    </li>
    </ul>
    <h3>Weight Space</h3>
    <ul>
    <li>Loss Landscape</li>
    <li>1 Datapoint → Full Model</li>
    </ul>
  </div>
  <div>
    <h3>Weight Space Learning</h3>
    <ul>
    <li>Meta-Learning</li>
    <li>Discriminate
        <ul>
        <li>Model Properties</li>
        <li>Performance</li>
        </ul>
    </li>
    <li>Generative
        <ul>
        <li>High-performing Model Weights</li>
        </ul>
    </li>
    </ul>
  </div>
</div>

---

## Problem
![bg height:17cm](image-2.png)
<!-- - Embedding the weight space
- Embed the dataset/task 
- Embed the corresponding result

Make use of Contrastive learning methods to combine these embeddings into a shared embedding space. Task/Weight/Result space to sample from and generate Weights conditioned on Result/Task/ 

Show the money-shot plot, input result vs output result.  -->