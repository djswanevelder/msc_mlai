## Dataset & Result conditioned Weight Generation 
This project contains all the relevant code for my MSc project

---

The fundamental goal is to have system which is capable of embedding a dataset $\mathcal{D}$, the trained neural weights $W$, and the corresponding results $R$ (training and test) into a shared embedding latent space. The goal is to use the shared space, to sample from the space  conditioned on a new unseen dataset and a specified results, and decode into real weights.  The architecture is kept constant for simplicity. 
$$ p (W \mid R, \mathcal{D},\text{Arch})
$$
The way in which this will be implemented requires a few building blocks.
- [Synthetic dataset generator](docs/data_gen.md)
- [Training pipeline and weight storage](docs/train_pipe.md)
- Weight Embedding
- Dataset Embedding
- Shared embedding projection

The goal of the project is to get the simplest implementation of the following up and running as fast as possible, and the evaluate the performance, then iteratively update the part of the pipeline to achieve better results. The goal is a graph, with input results (desired results) on the x-axis, measured results on the y. If the graph is somewhat linear, project is a success. 



