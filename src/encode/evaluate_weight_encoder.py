import torch
from torchvision import transforms
from PIL import Image
from src.encode.pca_method import PerParamPCAMapper, WeightSpaceAE, ResNet18WeightUtils, WeightVectorDataset, compare_model_outputs
from src.encode.encode_models import load_autoencoder, encode_resnet_model, decode_latent_to_resnet_model
from src.data_prep.prep_metadataset import load_artifact_metadata,standardize_class_name
from src.data_prep.resNet18 import prepare_data, ResNetClassifier
import ast
import torch.nn.functional as F
from typing import Tuple
import itertools
from src.shared_emb_space import predict_latent_vector
from tqdm import tqdm

def pearson_correlation(tensor1, tensor2, dim=0):
    
    mean1 = torch.mean(tensor1, dim=dim, keepdim=True)
    mean2 = torch.mean(tensor2, dim=dim, keepdim=True)
    centered1 = tensor1 - mean1
    centered2 = tensor2 - mean2
    
    cov = torch.sum(centered1 * centered2, dim=dim)
    std1_sq = torch.sum(centered1**2, dim=dim)
    std2_sq = torch.sum(centered2**2, dim=dim) # Corrected: 'dim' was accidentally set to 'torch.float32'
    
    corr = cov / (torch.sqrt(std1_sq * std2_sq) + 1e-8)
    return corr

def compare_models_with_permutation_test(model, reconstructed_model, val_loader) -> Tuple[float, float, float, float, float, float, float, float]:
    
    correct = agrees = correct_rec = total = rec_mse_sum = 0
    
    all_out = []
    all_out_rec = []
    all_pred = []
    all_pred_rec = []
    all_labels = []
    
    device = next(model.parameters()).device
    model.eval()
    reconstructed_model.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            out = model(images)
            out_rec = reconstructed_model(images)
            
            all_out.append(out.data.cpu())
            all_out_rec.append(out_rec.data.cpu())
            rec_mse_sum += torch.mean((out.data - out_rec.data)**2) * images.size(0)

            _, pred = torch.max(out.data, 1)
            _, pred_rec = torch.max(out_rec.data, 1)
            
            all_pred.append(pred.cpu())
            all_pred_rec.append(pred_rec.cpu())
            all_labels.append(labels.cpu())

            agrees += (pred == pred_rec).sum().item()
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            correct_rec += (pred_rec == labels).sum().item()
            
    full_out = torch.cat(all_out, dim=0)
    full_out_rec = torch.cat(all_out_rec, dim=0)
    full_pred = torch.cat(all_pred, dim=0)
    full_pred_rec = torch.cat(all_pred_rec, dim=0)
    full_labels = torch.cat(all_labels, dim=0)

    classes = [0, 1, 2]
    agreement = agrees / total
    reconstructed_accuracy = correct_rec / total

    best_agreement = agreement
    best_rec_accuracy = reconstructed_accuracy

    for p in itertools.permutations(classes):
        map_tensor = torch.tensor(p, dtype=torch.long)
        
        permuted_pred_rec = map_tensor[full_pred_rec]
        
        perm_agrees = (full_pred == permuted_pred_rec).sum().item()
        perm_agreement_ratio = perm_agrees / total
        best_agreement = max(best_agreement, perm_agreement_ratio)
        
        perm_correct_rec = (full_labels == permuted_pred_rec).sum().item()
        perm_rec_accuracy_ratio = perm_correct_rec / total
        best_rec_accuracy = max(best_rec_accuracy, perm_rec_accuracy_ratio)

    rec_mse = rec_mse_sum / total
    cos_sim = F.cosine_similarity(full_out, full_out_rec, dim=1).mean().item()
    corr_values = pearson_correlation(full_out.T, full_out_rec.T, dim=1)
    avg_correlation = corr_values.mean().item()
    
    original_accuracy = correct / total
    
    print(f"Reconstruction MSE (Output): {rec_mse:.4f}")
    print(f"Cosine Similarity (Output):  {cos_sim:.4f}")
    print(f"Output Correlation (Output): {avg_correlation:.4f}")
    print(f"--- Fixed Mapping ---")
    print(f"Prediction Agreement (Fixed): {100*agreement:.4f}%")
    print(f"Reconstructed Accuracy (Fixed): {100*reconstructed_accuracy:.4f}%")
    print(f"--- Best Permutation ---")
    print(f"Best Prediction Agreement: {100*best_agreement:.4f}%")
    print(f"Best Reconstructed Accuracy: {100*best_rec_accuracy:.4f}%")
    print(f"Original Accuracy: {100*original_accuracy:.4f}%")


    return rec_mse, agreement, original_accuracy, reconstructed_accuracy, cos_sim, avg_correlation, best_agreement, best_rec_accuracy


def encode_reconstruct_compare(model_file):
    model = ResNetClassifier(
        num_classes=3,
        optimizer_name='Adam',
        model_path = f'data/weights/selected/{model_file}'
    )

    weight_ae, mapper = load_autoencoder('data/dataset/pca_encoder.pth','data/dataset/')
    latent = encode_resnet_model(model, weight_ae, mapper)
    reconstructed_model = decode_latent_to_resnet_model(weight_ae, mapper, latent)


    _,val_loader = load_dataset('data/meta_data.csv', model_file)
    
    return compare_models_with_permutation_test(model, reconstructed_model, val_loader)
    # return compare_model_outputs(model, reconstructed_model)

def generate_model(classes,result):

    dataset_vector = f(classes)

    latent_vector = predict_latent_vector(
        model_path='data/trained_encoder_weights_best_val.pth',
        dataset_embedding=dataset_vector,
        validation_loss=result)

    weight_ae, mapper = load_autoencoder('data/dataset/pca_encoder.pth','data/dataset/')
    generated_model = decode_latent_to_resnet_model(weight_ae, mapper,latent_vector) 


    return generated_model

def load_dataset(input_file, model_file):
    meta_data_raw = load_artifact_metadata(input_file)
    for i, run in enumerate(meta_data_raw):
        if run['artifact_name'] == model_file:
            loss = run['val_loss']
            class1 = standardize_class_name(run['class1'])
            class2 = standardize_class_name(run['class2'])
            class3 = standardize_class_name(run['class3'])
            classes = [class1,class2,class3]

            mean = ast.literal_eval(run['calculated_mean'])
            std = ast.literal_eval(run['calculated_std'])
            print([class1,class2,class3,loss])
            break
    
    train_loader, val_loader, _, _, _ = prepare_data(
                                                        data_path='data/imagenet_data/',
                                                        classes_to_use=classes,
                                                        batch_size=32,
                                                        num_workers=6,
                                                        mean=mean,
                                                        std=std)  

    return train_loader,val_loader


if __name__ == '__main__':
    WEIGHT_FILES = ['ePZZH36doz3RPjmETdaTxo_67.pth', 'JSJkQKizxhwrFVpPtz3b3h_38.pth', 'F2WkFsEZ6GoPiLDQazAuq9_57.pth', 'Z9Wgk5q6PUhsN5bCCZbWzq_73.pth', 
                    'mBhkRxKsHnBYRPSr7Hbxsp_57.pth', 'WRQGZK2z2WgdChdDU5cBkd_44.pth', 'jaPLyma9tVfBQeCjsGr9MW_56.pth', '7bpJhWzwbdAFP8Wbz8vAi4_43.pth', 
                    'nKqPBdY6iMn9scYKxqZJct_13.pth', 'bgyUKNiPrvUTpJZAg4LXjq_18.pth', 'DprYVqsWJNKfyWfYpsrtrA_12.pth', '9RPMtNrv5wqTHcBpuqQxmJ_16.pth', 
                    '4qazTkXRYQa8xGbnDbcXeE_69.pth', 'U85ELXxJxEAry8EJCPu8wi_52.pth', 'eM2MkTS6x9uWUUWkBT4KbF_14.pth', 'hzcMnDq8QE47Ni96miQKUQ_10.pth', 
                    'gGSdGFYGZLNt4qh26LHgsw_12.pth', 'XX6QPKbXk8iv5v49B6hmjy_49.pth', 'Zhdme3mCqqZnYqms3VxLTi_44.pth', 'RGSKqWM4ZsRtxqptPjPhBu_18.pth', 
                    '5zUtmyD6sr64SCQDynuZpa_10.pth', 'ZZTCRjadANeZmhbAYA5D3e_15.pth', 'CjwsVETZ3MCLjcGUhganHA_41.pth', 'RLWSfuoAhxSACt2NEG6xCy_18.pth', 
                    'RrsFa56fpXy2RQE7ygioSe_42.pth', 'gRcwRQEdGqTWTeTJWiDmqd_21.pth', 'fhPDDsUyfJ6AYwtDwEWcSD_42.pth', 'kxNrByY452ceRV3sUiiffa_61.pth', 
                    'P3VegRLvhm5mCpzdTtRLxH_9.pth', 'TB47TAFZD8ZoWWfpRdvRwU_76.pth', 'MqBcBfKNR4vxu2Sras4JW7_48.pth', 'EeZ98m75VRPZrRHDxvejvd_43.pth', 
                    'WhhcYiZGkjb5Y5AN4Zo7RX_72.pth', 'Tp3W3AEKbt7oaqqzCKDp8Y_14.pth', 'fFUxenKM4yd6fopwn3qD5t_15.pth', 'ekDCZxna9Z5sVEHYpqiCeh_13.pth', 
                    'eMkQEiKCVCYf8XmNqVhPfb_29.pth', 'iB6XocRi3fUqdUmc7c3LdB_45.pth', 'dDyv43SjZ8d4WEoMTKiekr_14.pth', '4c6WfwmT4VNgdYvR5vzKD7_69.pth', 
                    '4Hwnhj4psanPs4Wdwxph9V_15.pth', 'gGSdGFYGZLNt4qh26LHgsw_48.pth', '532EHnro46vjXzHGgomhzD_12.pth', 'jy7iX9wbzT4m3oEbTDuJBq_9.pth', 
                    'eHnkzvkfjZuswc2t4v3s8w_16.pth', 'MppkioXhEa5xUt3BJKQymZ_16.pth', 'EPVKzUkxaqi4jaFAAoXrJt_16.pth', 'cpxK4Qcie8MRmBjFxdJ7um_18.pth', 
                    '5ZFPxQH8SunFTjNeNxAJJY_9.pth', 'TvcLonYMW8vsVurAeZqQJF_44.pth', '2jGPzZJrJNxU6ujivotyAi_27.pth', 'ePZZH36doz3RPjmETdaTxo_17.pth', 
                    'ekDCZxna9Z5sVEHYpqiCeh_80.pth', '5VPKMcijz5dd3cT5ojCG44_10.pth', 'JUoiGgEHtKqEJ48kXSW5EU_62.pth', '4qazTkXRYQa8xGbnDbcXeE_15.pth', 
                    '5vLswfJjve3JbjNebQCVex_9.pth', 'XMBs3TnEeo3Q9kzUjWaBcU_10.pth', 'eqgQz5SrLNgMiqPk2Db2ym_14.pth', 'Ba6gsyMXFaFEyBiAgLsaoo_15.pth', 
                    'mQarSuYK7vkRpJkJhscGVZ_9.pth', 'aPYWHvRkuUXh4qY8N4XEuE_75.pth', 'mWEy4mSg7FpQzUq92ZaPnj_18.pth', '9tMor4NNifAQ8xs2vvW666_60.pth', 'mpTyNaBs7GkBhBAbusY8rr_18.pth', 'JJ4jjJFkBNUBBNDWMGtggD_15.pth', 'oCxgYtPwQyUKojFq9GWqdA_18.pth', 'TB47TAFZD8ZoWWfpRdvRwU_18.pth', 
                    'Hfk6ebqjua8ve73r5UK7bX_59.pth', 'EEPneJg2zuDrrxGPwA5QoH_20.pth', 'WefoF4hVsuLP6vsKqHqtCn_49.pth', 'jJtsUMzrsAfu5NupBMutzw_13.pth', 
                    'fsprpwvqme6kgnxwkg6rtQ_59.pth', '4dWwLeJQRAj2DkBf39jANw_55.pth', 'VnaJ62XoxphbS9DfCsskTv_17.pth', 'bHCDjR8AP5EzNeGBaKqLRu_16.pth', 
                    'jWgjXd32hejMv5c3c7A9EY_50.pth', '6w4ztEb7JLALjgsaRHRcFg_11.pth', 'F2WkFsEZ6GoPiLDQazAuq9_11.pth', 'oEBE7furiAreUCPiLn3kJX_12.pth', 
                    ]
    
    all_results = []
    for weight in tqdm(WEIGHT_FILES):
    # for weight in WEIGHT_FILES:
        result = encode_reconstruct_compare(weight)
        all_results.append(result)
        print(result)
    print(f'{all_results}')

    import csv
    CSV_FILE_PATH = 'encoding_comparison_stats.csv'
    CSV_HEADINGS = [
        'rec_mse', 
        'agreement', 
        'original_accuracy', 
        'reconstructed_accuracy', 
        'cos_sim', 
        'avg_correlation', 
        'best_agreement', 
        'best_rec_accuracy'
    ]

    with open(CSV_FILE_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADINGS)
        writer.writerows(all_results)

