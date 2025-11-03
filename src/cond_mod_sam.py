from src.shared_emb_space import predict_latent_vector
from src.encode.encode_models import decode_latent_to_resnet_model, load_autoencoder




latent_vector = predict_latent_vector(
    model_path=ENCODER_MODEL_PATH,
    dataset_embedding=dataset_vector,
    validation_loss=target_val_loss)

weight_ae,mapper = load_autoencoder(CHECKPOINT_PATH)

decode_latent_to_resnet_model(
    weight_ae, mapper,
    latent_vector
) 