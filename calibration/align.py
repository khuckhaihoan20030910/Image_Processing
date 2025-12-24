import cv2
import numpy as np
import glob

# --- CẤU HÌNH ---
IMAGE_REF = "data/1/Basler_acA2500-14gm__22800124__20251114_112049766_0054.tiff"  # ảnh tham chiếu
IMAGE_TARGET_FOLDER = "data/turn4/"  # thư mục chứa các ảnh cần căn chỉnh
MAX_FEATURES = 5000
TOP_MATCHES = 300

# --- Hàm resize ảnh để hiển thị vừa màn hình ---
def shrink(img, max_w=1000):
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        return cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

# --- 1. Đọc ảnh tham chiếu ---
img_ref = cv2.imread(IMAGE_REF)
if img_ref is None:
    raise ValueError("Không đọc được ảnh tham chiếu")

img_ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

# --- 2. ORB để phát hiện feature trên ảnh tham chiếu ---
orb = cv2.ORB_create(MAX_FEATURES)
kp1, des1 = orb.detectAndCompute(img_ref_gray, None)

# --- 3. Duyệt các ảnh cần căn chỉnh ---
image_files = glob.glob(IMAGE_TARGET_FOLDER + "*.tiff")
image_files.sort()

for img_file in image_files:
    img_tgt = cv2.imread(img_file)
    if img_tgt is None:
        print(f"Không đọc được {img_file}, bỏ qua")
        continue

    img_tgt_gray = cv2.cvtColor(img_tgt, cv2.COLOR_BGR2GRAY)

    # --- 3a. Phát hiện feature ---
    kp2, des2 = orb.detectAndCompute(img_tgt_gray, None)
    if des2 is None or len(kp2) < 3:
        print(f"Không đủ feature trên {img_file}, bỏ qua")
        continue

    # --- 3b. Match feature ---
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:TOP_MATCHES]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # --- 3c. Tính Homography ---
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    h, w = img_ref.shape[:2]
    aligned = cv2.warpPerspective(img_tgt, H, (w, h))

    # --- 3d. Hiển thị ảnh ---
    img_ref_small = shrink(img_ref)
    aligned_small = shrink(aligned)

    # Hiển thị ảnh tham chiếu và ảnh căn chỉnh
    cv2.imshow("Ảnh chuẩn", img_ref_small)
    cv2.imshow("Ảnh căn chỉnh", aligned_small)

    # --- 3e. Overlay kiểm tra ---
    overlay = cv2.addWeighted(img_ref_small, 0.5, aligned_small, 0.5, 0)
    cv2.imshow("Overlay kiểm tra", overlay)

    print(f"Đã căn chỉnh xong {img_file}")

    # Chờ 1 giây trước khi chuyển sang ảnh tiếp theo
    if cv2.waitKey(5000) & 0xFF == 27:  # bấm ESC để thoát sớm
        break

cv2.destroyAllWindows()
print("Hoàn tất căn chỉnh và hiển thị tất cả ảnh.")
