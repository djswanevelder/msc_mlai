# Data Generation Plan
The plan is to make use of [ImageNet](https://image-net.org/download-images.php)'s variety of classes to create different datasets. A given combination of 3 classes will be considered a dataset. See [Heirarchy](https://observablehq.com/@mbostock/imagenet-hierarchy)
- First `download.py` is used to download all relevant classes
- Then a script will run generating a list of all the permuations which will be used as datasets. 

# Download.py
- Robustly create folder names for each classs
- Download specified classes (from `data.yaml`)
- Extract Images from .tar
