import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

def read_img(path):
    img = tiff.imread(str(path))
    if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16: img = (img / 256).astype(np.uint8)
    return img

def solve_strict_threshold(diff_image):
    """
    Tìm ngưỡng Otsu nhưng cộng thêm biên an toàn (Safety Margin)
    để né phần đuôi của Solder.
    """
    # 1. Lọc bỏ nền hoàn toàn để lấy dữ liệu Solder + Silk
    # Ngưỡng thấp (ví dụ 10) để loại bỏ nhiễu đen tuyệt đối
    valid_pixels = diff_image[diff_image > 10]
    
    if len(valid_pixels) == 0:
        return 50 # Fallback

    # 2. Chạy Otsu trên dữ liệu này
    otsu_thresh, _ = cv2.threshold(valid_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. TÍNH BIÊN AN TOÀN (QUAN TRỌNG)
    # Otsu thường nằm giữa thung lũng, nhưng Solder có thể "trườn" qua thung lũng đó.
    # Ta tính độ lệch chuẩn (std_dev) của phần dữ liệu bên dưới Otsu (chính là Solder)
    solder_pixels = valid_pixels[valid_pixels < otsu_thresh]
    
    if len(solder_pixels) > 0:
        # Tính độ phân tán của Solder
        std_dev = np.std(solder_pixels)
        # Công thức 3-Sigma: Đẩy ngưỡng lên 1.5 đến 2 lần độ lệch chuẩn để né Solder
        safety_buffer = int(std_dev * 1.5) 
    else:
        safety_buffer = 10 # Giá trị mặc định

    final_thresh = int(otsu_thresh + safety_buffer)
    
    return int(otsu_thresh), safety_buffer, final_thresh

def filter_small_blobs(mask, min_area=10):
    """
    Loại bỏ các đốm nhỏ (thường là sót lại của đỉnh mối hàn)
    """
    # Tìm các thành phần liên thông
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    output_mask = np.zeros_like(mask)
    
    # Duyệt qua các đối tượng (bỏ qua label 0 là nền)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Chỉ giữ lại đối tượng lớn hơn diện tích quy định
        # Silk thường là chữ nên diện tích sẽ lớn
        if area > min_area:
            output_mask[labels == i] = 255
            
    return output_mask

def process_strict_mode(image_path):
    print(f"--- STRICT MODE PROCESSING: {image_path} ---")
    original = read_img(image_path)
    
    # --- BƯỚC 1: DIFFERENCE MAP ---
    # Tăng kernel lên 61 để làm nền phẳng hơn nữa
    bg_blur = cv2.GaussianBlur(original, (61, 61), 0)
    diff = original.astype(np.float32) - bg_blur.astype(np.float32)
    diff[diff < 0] = 0
    diff_uint8 = diff.astype(np.uint8)
    
    # --- BƯỚC 2: TÍNH TOÁN NGƯỠNG THÔNG MINH ---
    otsu_val, buffer_val, strict_thresh = solve_strict_threshold(diff_uint8)
    
    print(f"[AUTO] Otsu Base: {otsu_val}")
    print(f"[AUTO] Safety Buffer: +{buffer_val} (Dựa trên độ phân tán Solder)")
    print(f"[AUTO] Final Threshold: {strict_thresh}")
    
    # --- BƯỚC 3: CẮT NGƯỠNG ---
    _, mask_raw = cv2.threshold(diff_uint8, strict_thresh, 255, cv2.THRESH_BINARY)
    
    # --- BƯỚC 4: LỌC HÌNH HỌC (GEOMETRIC FILTER) ---
    # Solder sót lại thường là chấm tròn nhỏ, Silk là nét liền
    # Xóa các đốm < 15 pixel (tùy chỉnh theo độ phân giải ảnh)
    mask_clean = filter_small_blobs(mask_raw, min_area=15)
    
    # --- HIỂN THỊ ---
    h, w = original.shape[:2]
    target_w = 700
    scale = target_w / w
    dim = (target_w, int(h * scale))

    # 1. Bản đồ chênh lệch
    view_diff = cv2.resize(diff_uint8, dim)
    view_diff = cv2.cvtColor(view_diff, cv2.COLOR_GRAY2BGR)
    cv2.putText(view_diff, "1. Diff Map", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    # 2. Mask trước khi lọc hạt nhỏ (Để thấy hiệu quả của ngưỡng Strict)
    view_raw = cv2.resize(mask_raw, dim)
    view_raw = cv2.cvtColor(view_raw, cv2.COLOR_GRAY2BGR)
    cv2.putText(view_raw, f"2. Strict Thresh ({strict_thresh})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    # 3. Mask sau khi lọc hạt (Clean)
    view_clean = cv2.resize(mask_clean, dim)
    view_clean = cv2.cvtColor(view_clean, cv2.COLOR_GRAY2BGR)
    cv2.putText(view_clean, "3. Area Filtered", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # 4. Overlay
    view_overlay = cv2.resize(cv2.cvtColor(original, cv2.COLOR_GRAY2BGR), dim)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view_overlay, contours, -1, (0, 0, 255), 1) # Vẽ màu đỏ cho nổi
    cv2.putText(view_overlay, f"Objects: {len(contours)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    final = np.hstack((view_diff, view_raw, view_clean, view_overlay))
    
    cv2.imshow("STRICT MODE SILK EXTRACTION", final)
    print("Nhấn phím bất kỳ để xem biểu đồ phân tích...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # --- VẼ BIỂU ĐỒ GIẢI THÍCH ---
    # Lấy dữ liệu pixel > 0 để vẽ
    data = diff_uint8[diff_uint8 > 10].ravel()
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=range(10, 256), color='gray', alpha=0.5, label='Pixel Distribution')
    
    plt.axvline(otsu_val, color='orange', linestyle='--', linewidth=2, label=f'Otsu ({otsu_val}) - Still has Solder')
    plt.axvline(strict_thresh, color='red', linewidth=3, label=f'Strict Thresh ({strict_thresh}) - Safe Zone')
    
    plt.title('Why Otsu fails & Why Strict Mode works')
    plt.xlabel('Intensity (Contrast)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    IMAGE_PATH = "data/1/Basler_acA2500-14gm__22800124__20251114_112049766_0054.tiff"
    process_strict_mode(IMAGE_PATH)