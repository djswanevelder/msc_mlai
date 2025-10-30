1.  Export Filtered Runs from wandb to `wandb_export.csv`
2. `python -m data.append_artifacts` to generate `wandb_w_artifacts.csv`
3. `python -m data.move` `weights` to `weights/selected` 
4. `wandb_w_artifacts.csv` -> `meta_data.csv` with `python -m data.meta_data`


5. Train PCA Weight Encoder
```
python -m src.encode.pca_method
```
6. Encoder the Models Latents
```
python -m src.encode.encode_models --model-dir data/weights/selected/ --checkpoint-path data/dataset/pca_encoder.pth --dataset-dir data/dataset --output-dir data/model_latents.pth
```
7. Generate Dataset (Class) Latents
```
python -m src.encode.encode_data
```
8. Prepare MetaDataset
```
python -m src.data_prep.prep_metadataset
```
9. Train Shared Encoder

```
python -m src.shared_emb_space
```

Generate Plots:


Loss
Embedding Table
Conditional model sampling
