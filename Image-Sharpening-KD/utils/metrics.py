from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import numpy as np

def compute_ssim(img1, img2):
    return ssim_metric(img1, img2, channel_axis=2)


