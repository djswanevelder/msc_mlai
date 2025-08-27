# Data Generation Plan
The plan is to make use of [ImageNet](https://image-net.org/download-images.php)'s variety of classes to create different datasets. A given combination of 3 classes will be considered a dataset. See [Heirarchy](https://observablehq.com/@mbostock/imagenet-hierarchy)
- `download.py` takes as input a list of classes, and downloads them
- `dataset_gen.py` randomly generate a `.csv` file with a list of possible permutations
    - Trianing Data for each permuation is added and stored in `sweep.csv`


# `download.py`
- Robustly create folder names for each classs
- Download specified classes (from `data.yaml`)
- Extract Images from .tar

# `dataset_gen.py`
- Generate random permuations 
- Generate random training hyperparamets
