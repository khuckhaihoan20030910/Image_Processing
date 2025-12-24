import cv2
import numpy as np
import tifffile as tiff
from skimage.filters import frangi

# ================== CẤU HÌNH ==================
FRANGI_SCALE = (1, 3)       # Hệ số quan trọng: Độ dày nét cần bắt (1px đến 3px)
TOPHAT_KERNEL = 40          # Kích thước lọc nền loang lổ
# ==============================================

def normalize_background(img):
    """
    Loại bỏ nền không đều màu bằng phép toán Top-Hat
    Giúp Frangi chỉ tập trung vào chi tiết nổi, không bị nhiễu bởi vùng sáng/tối cục bộ.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL))
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    norm = cv2.subtract(img, background)
    return norm

def get_ridge_map(img_gray):
    """
    Chỉ thực hiện Frangi Ridge Detection và trả về kết quả
    """
    # --- 1. Chuẩn hóa nền (Giữ lại bước này để Frangi hiệu quả hơn) ---
    norm = normalize_background(img_gray)

    # --- 2. Ridge / Line detection (Lõi thuật toán) ---
    # Giữ nguyên các tham số như code mẫu bạn gửi
    print("Đang chạy Frangi filter...")
    ridge = frangi(
        norm.astype(np.float32) / 255.0,
        scale_range=FRANGI_SCALE,
        scale_step=1,
        black_ridges=False
    )

    # Chuẩn hóa kết quả về 0-255 để hiển thị
    ridge_u8 = cv2.normalize(ridge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return ridge_u8

def show_small(name, img, scale=0.4):
    h, w = img.shape[:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, int(w * scale), int(h * scale))
    cv2.imshow(name, img)

# ================== MAIN ==================
if __name__ == "__main__":
    IMAGE_PATH = "data/1/Basler_acA2500-14gm__22800124__20251114_112049766_0054.tiff"

    # Đọc ảnh
    img = tiff.imread(IMAGE_PATH)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Lấy bản đồ Ridge
    ridge_result = get_ridge_map(img)

    # Hiển thị
    show_small("Original", img)
    show_small("Ridge Result (Frangi Only)", ridge_result)

    print("Đã xong. Nhấn phím bất kỳ để thoát.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()