import cv2
import numpy as np

# --- CẤU HÌNH TEST ---
MM_PER_PIXEL = 0.0125    # Giả sử: 1 pixel = 12.5 micromet
THRESH_SOLIDITY = 0.88   # Dưới mức này là bị sứt mẻ
THRESH_AREA_MAX = 5.0    # Trên mức này (mm2) nghi ngờ là bị dính (Bridged)

def create_simulation_image(shape=(600, 800)):
    """
    Tạo ảnh giả lập các tình huống: Chuẩn, Xoay, Sứt mẻ, Dính nhau.
    """
    img = np.zeros(shape, dtype=np.uint8)
    
    # 1. CHUẨN: Pad Tròn (BGA)
    # R=40px -> D=80px -> 1.0mm
    cv2.circle(img, (100, 150), 40, 255, -1)
    
    # 2. CHUẨN: Pad Chữ nhật (SMD 0805)
    # 60x100px -> 0.75x1.25mm
    cv2.rectangle(img, (200, 100), (260, 200), 255, -1)

    # 3. XOAY NGHIÊNG (Rotated)
    # Pad chữ nhật bị xoay 30 độ
    rect_rotated = ((400, 150), (50, 120), 30)
    box = np.int32(cv2.boxPoints(rect_rotated))
    cv2.drawContours(img, [box], 0, 255, -1)

    # 4. BỊ SỨT / MẺ (Chipped / Damaged)
    # Vẽ hình tròn, sau đó vẽ đè một hình tròn đen lên cạnh để tạo vết sứt
    cv2.circle(img, (150, 400), 50, 255, -1)      # Mối hàn gốc
    cv2.circle(img, (185, 400), 20, 0, -1)        # Vết sứt (màu đen)

    # 5. BỊ DÍNH / CHẬP (Bridged / Merged)
    # Hai pad nằm quá gần nhau nên bị dính lại
    cv2.circle(img, (400, 400), 40, 255, -1)
    cv2.circle(img, (450, 400), 40, 255, -1)      # Dính vào cái trên

    return img

def measure_solder_joints(mask, mm_per_pixel):
    """
    Đo lường sử dụng Convex Hull để khắc phục lỗi sứt mẻ.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    measurements = []

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < 50: continue 

        # --- A. TẠO CONVEX HULL (KHÔI PHỤC HÌNH DÁNG) ---
        hull = cv2.convexHull(cnt)
        hull_area_px = cv2.contourArea(hull)
        if hull_area_px <= 0: hull_area_px = area_px

        # --- B. TÍNH CHỈ SỐ HÌNH HỌC ---
        # Solidity: Độ đặc (Quan trọng để phát hiện sứt mẻ)
        solidity = float(area_px) / hull_area_px
        
        # Area: Tính theo Hull để bù đắp phần bị sứt
        area_mm2 = hull_area_px * (mm_per_pixel ** 2)

        # Size: Dùng minAreaRect trên Hull để đo chiều dài/rộng "lý thuyết"
        rect = cv2.minAreaRect(hull)
        (w_px, h_px) = rect[1]
        long_side = max(w_px, h_px) * mm_per_pixel
        short_side = min(w_px, h_px) * mm_per_pixel
        
        # Diameter: Đường kính tương đương (nếu coi là hình tròn)
        diameter_mm = np.sqrt(4 * area_mm2 / np.pi)

        # --- C. LẤY TÂM ---
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        measurements.append({
            'contour': cnt,
            'hull': hull,
            'center': (cX, cY),
            'solidity': round(solidity, 3),
            'area_mm2': round(area_mm2, 3),
            'size_mm': (round(long_side, 2), round(short_side, 2)),
            'diameter_mm': round(diameter_mm, 2)
        })
        
    return measurements

def visualize_results(img, measurements):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    print(f"{'Type Analysis':<20} | {'Solidity':<8} | {'Area(mm2)':<10} | {'Status'}")
    print("-" * 60)

    for m in measurements:
        cnt = m['contour']
        hull = m['hull']
        cx, cy = m['center']
        
        # 1. Vẽ Contour (Xanh lá - Thực tế)
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
        
        # 2. Vẽ Hull (Xanh dương - Khôi phục) - Chỉ vẽ nếu bị sứt
        if m['solidity'] < 1.0:
            cv2.drawContours(vis, [hull], -1, (255, 200, 0), 1)

        # 3. Phân tích & Đánh giá
        status_text = "OK"
        color_text = (0, 255, 0) # Green
        note = "Normal"

        # Case: BỊ DÍNH (Area quá to hoặc Shape kỳ lạ)
        if m['area_mm2'] > THRESH_AREA_MAX or (m['size_mm'][0] / m['size_mm'][1] > 3.0 and m['solidity'] < 0.7):
            status_text = "MERGED"
            color_text = (0, 0, 255) # Red
            note = "Bridged/Short"
        
        # Case: BỊ SỨT (Solidity thấp)
        elif m['solidity'] < THRESH_SOLIDITY:
            status_text = "CHIPPED"
            color_text = (0, 165, 255) # Orange
            note = "Missing Solder"

        # In log
        print(f"{note:<20} | {m['solidity']:<8} | {m['area_mm2']:<10} | {status_text}")

        # Vẽ Text lên ảnh
        # Dòng 1: Trạng thái (OK/BAD) + Độ đặc
        cv2.putText(vis, f"{status_text} (S:{m['solidity']})", (cx - 40, cy - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 2)
        
        # Dòng 2: Kích thước đã khôi phục (Hull Size)
        # Nếu bị sứt, kích thước này vẫn chính xác nhờ Hull
        size_str = f"{m['size_mm'][0]}x{m['size_mm'][1]}mm"
        cv2.putText(vis, size_str, (cx - 40, cy + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return vis

def main():
    print("--- SIMULATION: CHIPPED & BRIDGED JOINTS ---\n")
    
    # 1. Tạo ảnh giả lập
    mock_mask = create_simulation_image()
    
    # 2. Đo lường
    results = measure_solder_joints(mock_mask, MM_PER_PIXEL)
    
    # 3. Hiển thị
    final_view = visualize_results(mock_mask, results)
    
    cv2.imshow("Advanced Measurement Test", final_view)
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()