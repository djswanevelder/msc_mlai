# Data Generation Plan
The plan is to make use of [ImageNet](https://image-net.org/download-images.php)'s variety classes to create different datasets. A given combination of 3 classes will be considered a dataset. See [Heirarchy](https://observablehq.com/@mbostock/imagenet-hierarchy)
- First `download.py` needs to be used to download all of the required classe
- Then a script will run generating a list of all the permuations which will be used as datasets. 

# Download.py
- Robustly create file names
- Download specified classes (from `config.yaml`)
- Extract Images from .tar
