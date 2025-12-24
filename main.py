import cv2
import numpy as np
from lib import io, align, silk, solder, config as cfg

REF_PATH = "data/1/Basler_acA2500-14gm__22800124__20251114_112049766_0054.tiff"
TARGET_PATH = "data/turn1/Basler_acA2500-14gm__22800124__20251114_111339316_0045.tiff"
MM_PER_PIXEL = 0.029684 

# Biến toàn cục UI
ui_data = []      
clean_bg = None   

def mouse_callback(event, x, y, flags, param):
    """Xử lý sự kiện rê chuột (Interactive Hover)."""
    if event == cv2.EVENT_MOUSEMOVE:
        display = clean_bg.copy()
        found = False

        for item in ui_data:
            dist = cv2.pointPolygonTest(item['cnt'], (x, y), False)
            if dist >= 0: 
                found = True
                cv2.drawContours(display, [item['cnt']], -1, (0, 255, 255), 2)
                
                # Tooltip Info
                text = f"{item['w_mm']:.2f} x {item['h_mm']:.2f} mm"
                area_txt = f"Area: {item['area_mm']:.2f} mm2"
                
                # Vẽ hộp thông tin
                cv2.rectangle(display, (x + 10, y), (x + 180, y + 40), (50, 50, 50), -1)
                cv2.putText(display, text, (x + 20, y + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(display, area_txt, (x + 20, y + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                break 
        
        if not found:
            cv2.putText(display, f"Total: {len(ui_data)} | Hover to measure", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Smart Inspection", display)

def main():
    global clean_bg, ui_data

    img_ref = io.read_img(REF_PATH)
    img_tgt = io.read_img(TARGET_PATH)
    if img_ref is None or img_tgt is None: return

    print("[1] Aligning...")
    ref_silk_mask, _, _ = silk.extract_silk(img_ref)
    try: img_aligned = align.align_image(img_ref, img_tgt)
    except: img_aligned = img_tgt

    print("[2] Segmenting...")
    pcb_mask = solder.get_pcb_mask(img_aligned)
    # Raw mask: Chứa cả Solder và Silk dính nhau
    raw_mask = solder.segment_bright_objects(img_aligned, pcb_mask)

    print("[3] Reconstruction...")
    solder_final, solder_data = solder.filter_and_subtract_silk(raw_mask, ref_silk_mask)

    print(f"--> Found {len(solder_data)} joints.")

    # --- VISUALIZATION SETUP ---
    h, w = img_aligned.shape
    scale = 800 / h
    dim = (int(w * scale), 800)
    
    # Chuẩn bị ảnh nền cho cửa sổ chính
    clean_bg = cv2.cvtColor(cv2.resize(img_aligned, dim), cv2.COLOR_GRAY2BGR)
    
    # Chuẩn bị dữ liệu UI
    ui_data = []
    for item in solder_data:
        rect = cv2.minAreaRect(item['contour'])
        (w_px, h_px) = rect[1]
        cnt_view = (item['contour'] * scale).astype(np.int32)
        
        # Vẽ viền xanh mặc định
        cv2.drawContours(clean_bg, [cnt_view], -1, (0, 255, 0), 1)
        
        ui_data.append({
            'cnt': cnt_view,
            'w_mm': max(w_px, h_px) * MM_PER_PIXEL,
            'h_mm': min(w_px, h_px) * MM_PER_PIXEL,
            'area_mm': item['area'] * (MM_PER_PIXEL**2)
        })

    # --- HIỂN THỊ CÁC CỬA SỔ DEBUG (THEO YÊU CẦU) ---
    # 1. Ảnh gốc (Grayscale)
    view_raw_img = cv2.resize(img_aligned, dim)
    cv2.imshow("1. Raw Image (Aligned)", view_raw_img)

    # 2. Mask Silk (Dùng để cắt)
    view_silk = cv2.resize(ref_silk_mask, dim)
    cv2.imshow("2. Reference Silk Mask", view_silk)

    # 3. Raw Binary Mask (Solder + Silk dính nhau chưa xử lý)
    view_raw_mask = cv2.resize(raw_mask, dim)
    cv2.imshow("3. Raw Binary (Solder + Silk)", view_raw_mask)

    # 4. Cửa sổ chính (Interactive)
    window_name = "Smart Inspection"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Trigger lần đầu
    mouse_callback(cv2.EVENT_MOUSEMOVE, 0, 0, 0, 0)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()