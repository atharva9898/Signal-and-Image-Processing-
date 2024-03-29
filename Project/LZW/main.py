from LZW import LZW
import os
from PIL import Image
import numpy as np
from matplotlib.image import imread
import time
from skimage.metrics import peak_signal_noise_ratio as psnr

start_time = time.time()
compressor = LZW(os.path.join("Images","cup.tif"))
compressor.compress()

decompressor = LZW(os.path.join("CompressedFiles","cupCompressed.lzw"))
decompressor.decompress()

end_time = time.time()
computational_time = end_time - start_time

image_raw = imread(r"D:\SEM 4\SIP\Project\LZW\Images\cup.tif")
image_recon = imread(r"D:\SEM 4\SIP\Project\LZW\DecompressedFiles\cupDecompressed.tif")
original_image = np.array(image_raw)

recon_image = np.array(image_recon)

# Check if images have the same dimensions
#if original_image.shape != image_recon.shape:
#    raise ValueError("Input images must have the same dimensions.")
#
#psnr_value = psnr(image_raw, image_recon, data_range=image_raw.max() - image_raw.min())
#print("PSNR:", psnr_value)

## Calculate SSIM
#ssim_value = ssim(image_bw, image_recon, data_range=image_bw.max() - image_bw.min())
#print("SSI")
#

original_size = original_image.size
compressed_size = recon_image.size
compression_ratio = original_size / compressed_size
print("Compression Ratio:", compression_ratio)

print("Computation Complexity:", computational_time)




#def psnr(original, reconstructed):
#    """
#    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
#
#    Args:
#        original: The original image as a NumPy array.
#        reconstructed: The reconstructed image as a NumPy array.
#
#    Returns:
#        The PSNR value in dB.
#    """
#    # Ensure data type is float for MSE calculation
#    original = original.astype(np.float64)
#    reconstructed = reconstructed.astype(np.float64)
#
#    # Calculate Mean Squared Error (MSE)
#    mse = np.mean((original - reconstructed) ** 2)
#
#    # Handle zero MSE case (avoid division by zero)
#    if mse == 0:
#        return float('inf')
#
#    # Maximum possible pixel value (assuming 8-bit images)
#    max_pixel = 255.0
#
#    # Calculate PSNR (in dB)
#    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
#    
#    return psnr
#
#import cv2
#
## Convert to grayscale if needed (PSNR typically for grayscale images)
#original_img_gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
#reconstructed_img_gray = cv2.cvtColor(image_recon, cv2.COLOR_BGR2GRAY)
#
## Calculate PSNR
#psnr_value = psnr(original_img_gray, reconstructed_img_gray)
#
#print(f"PSNR of reconstructed image: {psnr_value:.2f} dB")