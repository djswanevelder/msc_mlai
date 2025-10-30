import torch
from torchvision import transforms
from PIL import Image
from src.encode.pca_method import PerParamPCAMapper, WeightSpaceAE, ResNet18WeightUtils
from src.encode.encode_models import load_autoencoder, encode_resnet_model, decode_latent_to_resnet_model
from src.data_prep.prep_metadataset import load_artifact_metadata,standardize_class_name
from src.data_prep.resNet18 import prepare_data, ResNetClassifier
def encode_reconstruct_classify(model_file):
    model = ResNetClassifier(
        num_classes=3,
        optimizer_name='Adam',
        model_path = f'data/weights/selected/{model_file}'
    )

    weight_ae, mapper = load_autoencoder('data/dataset/pca_encoder.pth','data/dataset/')
    latent = encode_resnet_model(model, weight_ae, mapper)
    reconstructed_model = decode_latent_to_resnet_model('data/dataset/pca_encoder.pth','data/dataset/', latent)


    train_loader,_ = load_dataset('data/meta_data.csv', model_file)
    
    # Classify the dataset using the model
    compare_model_outputs(train_loader,model, reconstructed_model)

def load_dataset(input_file, model_file):
    meta_data_raw = load_artifact_metadata(input_file)
    for i, run in enumerate(meta_data_raw):
        if run['artifact_name'] == model_file:
            loss = run['val_loss']
            class1 = standardize_class_name(run['class1'])
            class2 = standardize_class_name(run['class2'])
            class3 = standardize_class_name(run['class3'])
            classes = [class1,class2,class3]
            print([class1,class2,class3,loss])
            break
    
    train_loader, val_loader, _, _, _ = prepare_data(data_path='data/imagenet_data/',classes_to_use=classes,batch_size=32,num_workers=6)

    return train_loader,val_loader


def compare_model_outputs(dataset, original_model, reconstructed_model):

    for batch_idx, batch in enumerate(dataset):
        x, y = batch

        original_outputs = original_model(x)
        reconstructed_outputs = reconstructed_model(x)

        _, original_predicted = torch.max(original_outputs.data, 1)
        _, reconstructed_predicted = torch.max(reconstructed_outputs.data, 1)


        print('Ground Truth: ', y, 'Original Prediction: ', original_predicted, 'Reconstructed Prediction: ', reconstructed_predicted)
        if batch_idx == 0:
            break



if __name__ == '__main__':
    encode_reconstruct_classify('2FM3SRRFyeuu6K2uUHA5Q6_57.pth')
    