import cv2
import numpy as np
import glob
import os
from scipy.spatial import cKDTree

# ==========================================
# CẤU HÌNH
# ==========================================
FOLDER_PATH = "data/1/"  
FILE_EXTENSION = "*.tiff"
PITCH_REAL_MM = 2.54     # Khoảng cách chân chuẩn (mm)

def load_image_robust(path):
    """
    Đọc ảnh TIFF an toàn, xử lý vụ 3 kênh màu và 16-bit
    """
    # 1. Đọc nguyên bản (Unchanged)
    img_16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if img_16 is None:
        return None, None

    # 2. FIX LỖI: Chuyển 3 kênh thành 1 kênh nếu cần
    if len(img_16.shape) == 3:
        # Lấy kênh đầu tiên (Blue channel), giả định ảnh xám 3 kênh giống nhau
        img_16 = img_16[:, :, 0]
        
    # 3. Tạo bản 8-bit để hiển thị và Alignment
    img_8 = cv2.normalize(img_16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return img_16, img_8

def get_subpixel_centroid(img_16bit, contour):
    """Tính tâm Sub-pixel trên dữ liệu 16-bit"""
    mask = np.zeros_like(img_16bit, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    x, y, w, h = cv2.boundingRect(contour)
    roi_img = img_16bit[y:y+h, x:x+w]
    roi_mask = mask[y:y+h, x:x+w]
    
    # Chỉ tính trên pixel thuộc mask
    roi_input = cv2.bitwise_and(roi_img, roi_img, mask=roi_mask)
    
    M = cv2.moments(roi_input, binaryImage=False)
    if M["m00"] == 0: return None
        
    cx = x + M["m10"] / M["m00"]
    cy = y + M["m01"] / M["m00"]
    return (cx, cy)

def detect_and_measure(img_16, img_8):
    """Tìm lỗ và tính khoảng cách trung bình (pixels)"""
    blur = cv2.GaussianBlur(img_8, (5, 5), 0)
    
    # Otsu Thresholding
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50 or area > 10000: continue
        
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * area / (perimeter**2)
        
        if circularity < 0.6: continue 
        
        # Refine sub-pixel
        center = get_subpixel_centroid(img_16, c)
        if center: centers.append(center)
            
    if len(centers) < 2: return None
    
    # Tính Pitch bằng KDTree (tìm láng giềng gần nhất)
    pts = np.array(centers)
    tree = cKDTree(pts)
    dists = []
    
    for p in pts:
        d, _ = tree.query(p, k=2) 
        dists.append(d[1])
        
    dists = np.array(dists)
    median = np.median(dists)
    
    # Lọc nhiễu (chỉ lấy các khoảng cách gần median nhất)
    valid_dists = dists[np.abs(dists - median) < median * 0.15]
    
    if len(valid_dists) == 0: return None
    
    return np.mean(valid_dists)

def main_batch_processing():
    # 1. Lấy danh sách file
    search_path = os.path.join(FOLDER_PATH, FILE_EXTENSION)
    files = glob.glob(search_path)
    
    if not files:
        print(f" Không tìm thấy file nào trong: {search_path}")
        return
        
    print(f" Tìm thấy {len(files)} ảnh. Bắt đầu xử lý...")

    # 2. Xử lý ảnh đầu tiên để lấy mẫu (Template)
    first_path = files[0]
    img_16_ref, img_8_ref = load_image_robust(first_path)
    
    print(f" Hãy chọn vùng chứa lỗ trên ảnh đầu tiên: {os.path.basename(first_path)}")
    r = cv2.selectROI("Select Master ROI", img_8_ref, showCrosshair=True)
    cv2.destroyWindow("Select Master ROI")
    
    if r == (0,0,0,0): return

    rx, ry, rw, rh = int(r[0]), int(r[1]), int(r[2]), int(r[3])
    
    # Cắt template (dùng bản 8-bit để matching cho nhanh)
    template = img_8_ref[ry:ry+rh, rx:rx+rw]
    template_h, template_w = template.shape[:2]

    results_k = []
    
    # 3. Vòng lặp qua tất cả file
    for filepath in files:
        filename = os.path.basename(filepath)
        
        # Load ảnh
        img_16, img_8 = load_image_robust(filepath)
        if img_16 is None: continue
        
        # --- AUTO ALIGNMENT (Tìm vị trí mới của ROI) ---
        # Tìm template trong ảnh mới
        res = cv2.matchTemplate(img_8, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Nếu độ trùng khớp quá thấp -> Bỏ qua (ảnh lỗi hoặc mất PCB)
        if max_val < 0.6: 
            print(f" Bỏ qua {filename}: Không tìm thấy vùng ROI (Score: {max_val:.2f})")
            continue
            
        # Tọa độ ROI mới
        top_left = max_loc
        nx, ny = top_left
        
        # Cắt ROI trên ảnh 16-bit tại vị trí mới
        roi_16 = img_16[ny:ny+template_h, nx:nx+template_w]
        roi_8 = img_8[ny:ny+template_h, nx:nx+template_w] # Chỉ dùng debug nếu cần
        
        # --- ĐO ĐẠC ---
        avg_pitch_px = detect_and_measure(roi_16, roi_8)
        
        if avg_pitch_px:
            k = PITCH_REAL_MM / avg_pitch_px
            results_k.append(k)
            print(f" {filename}: Pitch = {avg_pitch_px:.2f} px -> K = {k:.6f} mm/px")
        else:
            print(f" {filename}: Không detect được lỗ nào.")

    # 4. TỔNG HỢP KẾT QUẢ
    if results_k:
        k_final = np.mean(results_k)
        k_std = np.std(results_k)
        
        print("\n" + "="*40)
        print("   KẾT QUẢ HIỆU CHUẨN TOÀN BỘ FOLDER")
        print("="*40)
        print(f"Số lượng ảnh hợp lệ: {len(results_k)} / {len(files)}")
        print(f"K trung bình:        {k_final:.8f} mm/pixel")
        print(f"Độ ổn định (Std):    {k_std:.8f}")
        print(f"Sai số ước tính:     ±{k_std/k_final*100:.4f}%")
        print("="*40)
    else:
        print("\n Không thu được dữ liệu nào.")

if __name__ == "__main__":
    main_batch_processing()