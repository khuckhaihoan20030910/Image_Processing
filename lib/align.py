# lib/align.py
import cv2
import numpy as np

# Cấu hình riêng cho việc Align
MAX_FEATURES = 5000
TOP_MATCHES = 300

def align_image(img_ref, img_target):
    """
    Căn chỉnh img_target khớp theo img_ref dùng ORB + Homography
    """
    # 1. Detect Features
    orb = cv2.ORB_create(MAX_FEATURES)
    kp1, des1 = orb.detectAndCompute(img_ref, None)
    kp2, des2 = orb.detectAndCompute(img_target, None)

    if des2 is None or len(kp2) < 3:
        print("Cảnh báo: Không đủ đặc trưng để căn chỉnh!")
        return img_target # Trả về ảnh gốc nếu không căn chỉnh được

    # 2. Match
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:TOP_MATCHES]

    # 3. Find Homography & Warp
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    h, w = img_ref.shape[:2]
    
    aligned = cv2.warpPerspective(img_target, H, (w, h))
    return aligned