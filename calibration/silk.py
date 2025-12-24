import cv2
import numpy as np
import tifffile as tiff
from skimage.filters import frangi
from skimage.morphology import reconstruction, remove_small_objects
from skimage.measure import label, regionprops

# ================== CẤU HÌNH TINH CHỈNH (TUNING) ==================

# 1. Cấu hình Threshold (Hạt giống)
STRICT_BUFFER = 1.5         
SEED_MIN_SIZE = 15          

# 2. Cấu hình Ridge (Vỏ bọc)
FRANGI_SCALE = (1, 3)       
FRANGI_BETA = 0.5           
RIDGE_LOW_THRESH = 40       
RIDGE_MIN_SIZE = 20         

# [MỚI - QUAN TRỌNG] Cấu hình phân loại Solder vs Silk tròn
MIN_ECCENTRICITY = 0.8      # Dưới mức này bị coi là "Tròn"
MAX_SOLDER_DIA = 20        # [MỚI] Đường kính tối đa của Solder (pixel). 
                            # Tròn mà nhỏ hơn số này -> Xóa. 
                            # Tròn mà to hơn số này -> Giữ.

# 3. Cấu hình chung
TOPHAT_KERNEL = 15          
FINAL_MIN_SIZE = 50         

# ==================================================================

def read_img(path):
    try:
        img = tiff.imread(str(path))
    except Exception as e:
        print(f"Lỗi đọc ảnh: {e}")
        return None
    if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16: img = (img / 256).astype(np.uint8)
    return img

def clean_noise(mask, min_size):
    if np.sum(mask) == 0: return mask
    clean = remove_small_objects(mask > 0, min_size=min_size)
    return (clean * 255).astype(np.uint8)

def filter_smart_solder(mask, min_eccentricity, max_solder_diameter):
    """
    [THUẬT TOÁN MỚI] Lọc thông minh:
    Giữ lại đối tượng NẾU:
    1. Nó là nét mảnh (Eccentricity cao)
       HOẶC
    2. Nó là hình tròn NHƯNG kích thước lớn (Đường kính > max_solder_diameter)
    """
    if np.sum(mask) == 0: return mask
    
    label_img = label(mask)
    regions = regionprops(label_img)
    
    out_mask = np.zeros_like(mask)
    
    print(f"--- Phân tích {len(regions)} đối tượng trong Ridge ---")
    
    for props in regions:
        # Điều kiện 1: Là nét (không phải khối tròn)
        is_line = props.eccentricity >= min_eccentricity
        
        # Điều kiện 2: Là khối tròn, nhưng kích thước TO hơn Solder
        # equivalent_diameter: đường kính của hình tròn có cùng diện tích
        is_big_circle = (props.eccentricity < min_eccentricity) and \
                        (props.equivalent_diameter > max_solder_diameter)
        
        # Logic: Giữ lại nếu là Line HOẶC là Big Circle
        if is_line or is_big_circle:
            minr, minc, maxr, maxc = props.bbox
            out_mask[minr:maxr, minc:maxc] = np.maximum(
                out_mask[minr:maxr, minc:maxc], 
                mask[minr:maxr, minc:maxc] * props.image
            )
            
    return out_mask

def get_marker_from_threshold(img_gray):
    # (Giữ nguyên như cũ)
    bg_blur = cv2.GaussianBlur(img_gray, (61, 61), 0)
    diff = img_gray.astype(np.float32) - bg_blur.astype(np.float32)
    diff[diff < 0] = 0
    diff_uint8 = diff.astype(np.uint8)

    valid_pixels = diff_uint8[diff_uint8 > 10]
    if len(valid_pixels) == 0: 
        return np.zeros_like(img_gray), 0, diff_uint8
    
    otsu_val, _ = cv2.threshold(valid_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    solder_pixels = valid_pixels[valid_pixels < otsu_val]
    if len(solder_pixels) > 0:
        std_dev = np.std(solder_pixels)
        safety_buffer = int(std_dev * STRICT_BUFFER)
    else:
        safety_buffer = 10

    strict_thresh = int(otsu_val + safety_buffer)
    _, marker_mask = cv2.threshold(diff_uint8, strict_thresh, 255, cv2.THRESH_BINARY)
    marker_clean = clean_noise(marker_mask, SEED_MIN_SIZE)
    return marker_clean, strict_thresh, diff_uint8

def get_mask_from_ridge(img_gray):
    # (Cập nhật hàm lọc mới)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL))
    background = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
    norm = cv2.subtract(img_gray, background)

    print("Đang chạy Frangi Filter...")
    ridge = frangi(
        norm.astype(np.float32)/255.0, 
        scale_range=FRANGI_SCALE, 
        beta=FRANGI_BETA, 
        black_ridges=False
    )
    ridge_u8 = cv2.normalize(ridge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    _, mask_loose = cv2.threshold(ridge_u8, RIDGE_LOW_THRESH, 255, cv2.THRESH_BINARY)
    mask_clean_noise = clean_noise(mask_loose, RIDGE_MIN_SIZE)
    
    # [THAY ĐỔI Ở ĐÂY] Dùng hàm lọc thông minh mới
    print(f"Lọc Solder: Eccentricity < {MIN_ECCENTRICITY} VÀ Diameter < {MAX_SOLDER_DIA}px")
    mask_smart = filter_smart_solder(mask_clean_noise, MIN_ECCENTRICITY, MAX_SOLDER_DIA)
    
    return mask_smart, ridge_u8

def solve_hybrid(image_path):
    print(f"--- BẮT ĐẦU XỬ LÝ (SMART SIZE FILTER): {image_path} ---")
    img = read_img(image_path)
    if img is None: return

    # Bước 1
    marker, strict_val, diff_map = get_marker_from_threshold(img)
    print(f"[1] Marker OK")

    # Bước 2
    mask_envelope, ridge_vis = get_mask_from_ridge(img)
    print(f"[2] Ridge Mask OK")

    # Bước 3
    print("[3] Reconstruction...")
    seed = cv2.bitwise_and(marker, mask_envelope)
    final_reconstructed = reconstruction(seed, mask_envelope, method='dilation')
    final_mask = final_reconstructed.astype(np.uint8)

    # Bước 4
    final_clean = clean_noise(final_mask, FINAL_MIN_SIZE)

    # Hiển thị
    h, w = img.shape[:2]
    display_width = 600
    scale = display_width / w
    dim = (display_width, int(h * scale))

    def to_bgr_resized(gray_img):
        return cv2.cvtColor(cv2.resize(gray_img, dim), cv2.COLOR_GRAY2BGR)

    v1 = to_bgr_resized(marker)
    cv2.putText(v1, "1. Clean Seed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    v2 = to_bgr_resized(mask_envelope)
    cv2.putText(v2, f"2. Mask (Dia > {MAX_SOLDER_DIA})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    v3 = to_bgr_resized(final_clean)
    cv2.putText(v3, "3. FINAL RESULT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    v4 = to_bgr_resized(img)
    mask_for_view = cv2.resize(final_clean, dim)
    cnts, _ = cv2.findContours(mask_for_view, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(v4, cnts, -1, (0, 255, 0), 1)
    
    combined = np.vstack((np.hstack((v1, v2)), np.hstack((v3, v4))))
    cv2.imshow("HYBRID - SMART DIAMETER", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    IMAGE_PATH = "data/1/Basler_acA2500-14gm__22800124__20251114_112049766_0054.tiff"
    solve_hybrid(IMAGE_PATH)