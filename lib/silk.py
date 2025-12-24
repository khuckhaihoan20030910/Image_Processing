import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import reconstruction
from skimage.measure import label, regionprops
# Import file cấu hình và tiện ích trong cùng gói lib
from . import config as cfg
from . import io

def get_marker_seed(img_gray):
    """Bước 1: Tạo Hạt giống (Threshold)"""
    # Xử lý nền
    bg_blur = cv2.GaussianBlur(img_gray, (61, 61), 0)
    diff = img_gray.astype(np.float32) - bg_blur.astype(np.float32)
    diff[diff < 0] = 0
    diff_uint8 = diff.astype(np.uint8)

    # Tính Otsu
    valid_pixels = diff_uint8[diff_uint8 > 10]
    if len(valid_pixels) == 0: 
        return np.zeros_like(img_gray), 0
    
    otsu_val, _ = cv2.threshold(valid_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Tính đệm an toàn
    solder_pixels = valid_pixels[valid_pixels < otsu_val]
    if len(solder_pixels) > 0:
        std_dev = np.std(solder_pixels)
        safety = int(std_dev * cfg.STRICT_BUFFER)
    else:
        safety = 10

    strict_thresh = int(otsu_val + safety)
    _, marker_mask = cv2.threshold(diff_uint8, strict_thresh, 255, cv2.THRESH_BINARY)
    
    # Lọc rác
    return io.clean_noise(marker_mask, cfg.SEED_MIN_SIZE), strict_thresh

def filter_smart_solder(mask):
    """Bước 2b: Lọc thông minh (Dựa trên Eccentricity và Diameter)"""
    if np.sum(mask) == 0: return mask
    
    label_img = label(mask)
    regions = regionprops(label_img)
    out_mask = np.zeros_like(mask)
    
    for props in regions:
        # 1. Giữ nếu là NÉT (không tròn)
        is_line = props.eccentricity >= cfg.MIN_ECCENTRICITY
        
        # 2. Giữ nếu là TRÒN nhưng TO (Silk tròn > MAX_SOLDER_DIA)
        is_big_circle = (props.eccentricity < cfg.MIN_ECCENTRICITY) and \
                        (props.equivalent_diameter > cfg.MAX_SOLDER_DIA)
        
        if is_line or is_big_circle:
            minr, minc, maxr, maxc = props.bbox
            out_mask[minr:maxr, minc:maxc] = np.maximum(
                out_mask[minr:maxr, minc:maxc], 
                mask[minr:maxr, minc:maxc] * props.image
            )
    return out_mask

def get_ridge_envelope(img_gray):
    """Bước 2a: Tạo Vỏ bọc (Ridge)"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.TOPHAT_KERNEL, cfg.TOPHAT_KERNEL))
    background = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
    norm = cv2.subtract(img_gray, background)

    ridge = frangi(
        norm.astype(np.float32)/255.0, 
        scale_range=cfg.FRANGI_SCALE, 
        beta=cfg.FRANGI_BETA, 
        black_ridges=False
    )
    ridge_u8 = cv2.normalize(ridge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    _, mask_loose = cv2.threshold(ridge_u8, cfg.RIDGE_LOW_THRESH, 255, cv2.THRESH_BINARY)
    
    # Lọc rác nhỏ trước
    mask_clean_noise = io.clean_noise(mask_loose, cfg.RIDGE_MIN_SIZE)
    # Lọc Solder thông minh
    mask_smart = filter_smart_solder(mask_clean_noise)
    
    return mask_smart

# --- HÀM CHÍNH (Đã đặt đúng tên là extract_silk) ---
def extract_silk(img_gray):
    """
    Hàm kết hợp (Hybrid) để tách Silk.
    Trả về: (Mask kết quả, Marker để debug, Envelope để debug)
    """
    marker, _ = get_marker_seed(img_gray)
    envelope = get_ridge_envelope(img_gray)
    
    seed = cv2.bitwise_and(marker, envelope)
    rec = reconstruction(seed, envelope, method='dilation').astype(np.uint8)
    final = io.clean_noise(rec, cfg.FINAL_MIN_SIZE)
    
    return final, marker, envelope