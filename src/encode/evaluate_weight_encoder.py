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

    # generate latent 
    latent = []
    generated_model = decode_latent_to_resnet_model('data/dataset/pca_encoder.pth','data/dataset/', latent)

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
    WEIGHT_FILES = [
        '2qrZZxz7s2pw8jZpc3uQ6w_13.pth', 'bFFucNm8wrHoUpYRyVdcvV_47.pth', 'kQNN9BKxw27icC5FQgDoYL_49.pth', 'NtcyLyyLdAqGm5XLreFTRB_11.pth', 'TvcLonYMW8vsVurAeZqQJF_44.pth',
        '2qrZZxz7s2pw8jZpc3uQ6w_60.pth', 'ckcvGPRvJPE5vJA3Gh4poB_14.pth', 'Gn2eo8JNUQ8nW9yhWpyqXx_9.pth', 'M8DQQZgH2bYxrwSNDUub9h_13.pth', 'NtcyLyyLdAqGm5XLreFTRB_51.pth', 'VfyDtshcsAw3uvrtJunLyy_15.pth',
        '4aDxBFkc4qtLwkjnYsADzW_17.pth', 'ckcvGPRvJPE5vJA3Gh4poB_78.pth', 'gzQj8aw7SsdLTAcr8NKHcD_11.pth', 'M8DQQZgH2bYxrwSNDUub9h_30.pth', 'P2LArrtfTHkt5zwvjHBUbJ_15.pth', 'VfyDtshcsAw3uvrtJunLyy_23.pth',
        '4aD  xBFkc4qtLwkjnYsADzW_35.pth', 'dXnK663qp9qfCHsjsZcV35_13.pth', 'gzQj8aw7SsdLTAcr8NKHcD_70.pth', 'mBhkRxKsHnBYRPSr7Hbxsp_17.pth', 'P2LArrtfTHkt5zwvjHBUbJ_49.pth', 'VosYUcDPcYy986N7bBMQkQ_17.pth',
        '5pb9UcfxwrLcC6kmvYyt5o_12.pth', 'dXnK663qp9qfCHsjsZcV35_34.pth', 'h7CTx9uA6t7eE3Znm5TcWD_18.pth', 'mBhkRxKsHnBYRPSr7Hbxsp_57.pth', 'PkQcWcwebYqfpxQ9AQ3nKX_11.pth', 'VosYUcDPcYy986N7bBMQkQ_46.pth',
        '5pb9UcfxwrLcC6kmvYyt5o_24.pth', 'eqtz4uyzM2W5fVkywXGds7_17.pth', 'h7CTx9uA6t7eE3Znm5TcWD_45.pth', 'mLCMxTPh7bxSrbF7exvVkN_15.pth', 'PkQcWcwebYqfpxQ9AQ3nKX_25.pth', 'WuJCb47eH7eXnQiNB2hQoJ_13.pth',
        '6GuNTk3ZGN76raPSvhmyAc_73.pth', 'eqtz4uyzM2W5fVkywXGds7_60.pth', 'hDJ2hTFHJe8Q3F4c56jkw9_17.pth', 'mLCMxTPh7bxSrbF7exvVkN_59.pth', 'SPDgGD3J4iMdEX49TaiYkg_11.pth', 'WuJCb47eH7eXnQiNB2hQoJ_46.pth',
        '6GuNTk3ZGN76raPSvhmyAc_9.pth', 'F9JWeacPyYrXJFc5AWikxK_15.pth', 'hDJ2hTFHJe8Q3F4c56jkw9_50.pth', 'mpTyNaBs7GkBhBAbusY8rr_18.pth', 'SPDgGD3J4iMdEX49TaiYkg_72.pth', 'Yu9Gmhri3pmiR3aMqGKQrQ_15.pth',
        '7bpJhWzwbdAFP8Wbz8vAi4_13.pth', 'F9JWeacPyYrXJFc5AWikxK_33.pth', 'hHLjAHiYrvrBrpmpAcfdj6_12.pth', 'mpTyNaBs7GkBhBAbusY8rr_20.pth', 'SQjYoJGfHjrBKuZEqcncqb_15.pth', 'Yu9Gmhri3pmiR3aMqGKQrQ_80.pth',
        '7bpJhWzwbdAFP8Wbz8vAi4_43.pth', 'fFUxenKM4yd6fopwn3qD5t_15.pth', 'hHLjAHiYrvrBrpmpAcfdj6_51.pth', 'MqBcBfKNR4vxu2Sras4JW7_11.pth', 'SQjYoJGfHjrBKuZEqcncqb_75.pth', 'YXjpantVgur78w6gQ8Ya2B_12.pth',
        'AhxyZLpDPXN2LgVVfGqM6s_57.pth', 'fFUxenKM4yd6fopwn3qD5t_27.pth', 'ihji9nNheYJNWTijuSRZRQ_16.pth', 'MqBcBfKNR4vxu2Sras4JW7_48.pth', 'ThEu6qa5mh5rtdRkBAAAMP_11.pth', 'YXjpantVgur78w6gQ8Ya2B_38.pth',
        'AhxyZLpDPXN2LgVVfGqM6s_9.pth', 'fhPDDsUyfJ6AYwtDwEWcSD_12.pth', 'ihji9nNheYJNWTijuSRZRQ_72.pth', 'mWA9NSt3oNbQAuYTkiTXVU_54.pth', 'ThEu6qa5mh5rtdRkBAAAMP_77.pth', 
        'B6AkqyA5UxUr7inszj3VQD_15.pth', 'fhPDDsUyfJ6AYwtDwEWcSD_42.pth', 'jtHSyAnUaKLpEZFbSyQWJi_17.pth', 'mWA9NSt3oNbQAuYTkiTXVU_9.pth', 'TjReEM9t34rLtgT8od2fPj_17.pth', 
        'B6AkqyA5UxUr7inszj3VQD_66.pth', 'gDiwxdfc2TdGvRJAUzWmku_12.pth', 'jtHSyAnUaKLpEZFbSyQWJi_51.pth', 'NTC953tsivv7rvRbPveqVv_18.pth', 'TjReEM9t34rLtgT8od2fPj_62.pth', 
        'bFFucNm8wrHoUpYRyVdcvV_18.pth', 'gDiwxdfc2TdGvRJAUzWmku_30.pth', 'kQNN9BKxw27icC5FQgDoYL_13.pth', 'NTC953tsivv7rvRbPveqVv_46.pth', 'TvcLonYMW8vsVurAeZqQJF_13.pth'
    ]
    
   # Initialize accumulators for the metrics we want to average
    total_acc_diff = 0.0
    total_cos_sim = 0.0
    total_avg_correlation = 0.0
    num_models = 0 # Counter for the number of models processed

    print("Starting model reconstruction and comparison...")

    for weight in WEIGHT_FILES:
        try:
            # The compare_model_outputs function should return the metrics
            # Based on your internal compare_models_with_permutation_test, 
            # let's assume it returns: 
            # (rec_mse, agreement, original_accuracy, reconstructed_accuracy, cos_sim, avg_correlation, best_agreement, best_rec_accuracy)
            
            # NOTE: Your original code calls compare_model_outputs, 
            # but the provided compare_models_with_permutation_test returns 8 values.
            # I'll update the variable names to match the 8-tuple from the function you provided.
            
            # You should ensure 'compare_model_outputs' is defined and returns these 8 values, 
            # or simply change 'encode_reconstruct_compare' to call the function you provided:
            # return compare_models_with_permutation_test(model, reconstructed_model, val_loader)
            
            # Assuming 'encode_reconstruct_compare' calls 'compare_models_with_permutation_test' and returns the 8 values:
            rec_mse, agreement, original_accuracy, reconstructed_accuracy, cos_sim, avg_correlation, best_agreement, best_rec_accuracy = encode_reconstruct_compare(weight)

            # Calculate the Accuracy Difference (Original - Best Reconstructed)
            # The accuracy difference is typically calculated using the best-case (permuted) reconstructed accuracy
            acc_diff = original_accuracy - best_rec_accuracy
            
            # Accumulate the results
            total_acc_diff += acc_diff
            total_cos_sim += cos_sim
            total_avg_correlation += avg_correlation
            num_models += 1
            
            print(f"--- Results for {weight} ---")
            print(f"Accuracy Diff (Original - Best Rec): {acc_diff:.4f}")
            print(f"Cosine Sim: {cos_sim:.4f}")
            print(f"Correlation: {avg_correlation:.4f}\n")
            
        except Exception as e:
            print(f"Skipping model {weight} due to an error: {e}")


    # Calculate and print the final averages
    if num_models > 0:
        avg_acc_diff = total_acc_diff / num_models
        avg_cos_sim = total_cos_sim / num_models
        avg_correlation = total_avg_correlation / num_models
        
        print("\n" + "="*50)
        print(f"✨ Final Averages Over {num_models} Models ✨")
        print("="*50)
        print(f"Average Accuracy Difference (Original - Best Rec): {avg_acc_diff:.4f}")
        print(f"Average Cosine Similarity (Output): {avg_cos_sim:.4f}")
        print(f"Average Correlation (Output): {avg_correlation:.4f}")
        print("="*50)
    else:
        print("\nNo models were successfully processed to calculate averages.")