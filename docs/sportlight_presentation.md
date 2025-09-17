## Description
This should be again a stress-free exercise: 3 slides for  
  
1- the title  
2- the context (try to relate to something we know or have seen during the year - what’s the area of the problem?)  
3- state your particular problem.  
  
5 mins + 5 mins questions. It’ll be nice to see what everybody is working on.  
  
Very important: send me your slides up to 2pm Friday so we have all the presentations on one computer. The schedule is tight.

[[Spotlight Slides]]
[[PMR - Presentation]]

## Title
A Contrastive Learning Approach to Weight-Space Learning
25205269 | DJ Swanevelder
Supervisor: Ruan van der Merwe
## Context
(try to relate to something we know or have seen during the year - what’s the area of the problem?)  
- Contrastive Learning
	- A measure of similarity 
	- Meaningful representations in a new space, which reflects the similarity metric 
	- Embedding/NLP - represent semantic relationships of words 
	- Negative Sampling
- Weight-Space
	- Loss-landscape, the loss distribution over the weight space. 
	- Simply all the possible values and combinations of values of the weights of a model.
	- A single point in the weight space, represents a full model
- Weight-Space Learning
	- Simply treats a point in the weight-space as a single data point.
	- Often Meta-learning, cocerned with Methods that model other models. 
	- Downstream Tasks:
		- Discriminative
			- Predicting a model's performance or properties based on only weights
			- Training properties
				- Epochs
				- Optimizer
				- Learning Rate
				- Dataset Size
			- Performance:
				- Generalization Gap
				- Test performance
		- Generative
			- New Neural Network weights
## Problem
- Embedding the weight space
- Embed the dataset/task 
- Embed the corresponding result

Make use of Contrastive learning methods to combine these embeddings into a shared embedding space. Task/Weight/Result space to sample from and generate Weights conditioned on Result/Task/ 

Show the money-shot plot, input result vs output result. 


