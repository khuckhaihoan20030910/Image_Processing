# lib/io.py
import cv2
import numpy as np
import tifffile as tiff
from skimage.morphology import remove_small_objects

def read_img(path):
    """Đọc ảnh và chuyển đổi sang uint8 grayscale chuẩn"""
    try:
        img = tiff.imread(str(path))
    except Exception as e:
        print(f"Lỗi đọc ảnh: {e}")
        return None

    if img.ndim == 3: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16: 
        img = (img / 256).astype(np.uint8)
    return img

def clean_noise(mask, min_size):
    """Hàm phụ trợ để lọc rác nhỏ bằng skimage"""
    if np.sum(mask) == 0: return mask
    # remove_small_objects nhận input là bool
    clean = remove_small_objects(mask > 0, min_size=min_size)
    return (clean * 255).astype(np.uint8)

def show_result(name, img, width=800):
    """Hàm hiển thị ảnh resize nhanh"""
    h, w = img.shape[:2]
    scale = width / w
    dim = (width, int(h * scale))
    resized = cv2.resize(img, dim)
    cv2.imshow(name, resized)