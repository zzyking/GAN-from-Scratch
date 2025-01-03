from torch.nn import functional as F
import torch
import torch.nn as nn
import numpy as np
import scipy


def get_inception_features(images, inception_model, batch_size=32, device='cuda'):
    features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + 32].to(device)
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False).to(device)
            batch_features = inception_model(batch)
            features.append(batch_features.cpu().numpy())
        torch.cuda.empty_cache()
    return np.concatenate(features, axis=0)

def compute_real_features(test_loader, inception_model, device):
    real_features = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False).to(device)
            features = inception_model(imgs)
            real_features.append(features.cpu().numpy())
        torch.cuda.empty_cache()
    return np.concatenate(real_features, axis=0)

# FID 计算

def calculate_fid(real_features, generated_features):
    mu1 = np.mean(generated_features, axis=0)
    mu2 = np.mean(real_features, axis=0)
    
    sigma1 = np.cov(generated_features, rowvar=False) + 1e-5 * np.eye(generated_features.shape[1])
    sigma2 = np.cov(real_features, rowvar=False) + 1e-5 * np.eye(real_features.shape[1])
    
    diff = mu1 - mu2
    diff_sq = np.dot(diff, diff.T)
    
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    cov_trace = np.trace(sigma1 + sigma2 - 2*covmean)
    
    fid = diff_sq + cov_trace
    return fid