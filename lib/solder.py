import cv2
import numpy as np
from skimage.morphology import reconstruction
from . import config as cfg
from . import io 

def get_pcb_mask(img):
    if img.ndim == 3: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: gray = img.copy()
    
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    thresh_val, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    h, w = mask.shape
    corners = [mask[0:20, 0:20], mask[0:20, w-20:w], mask[h-20:h, 0:20], mask[h-20:h, w-20:w]]
    if np.mean([np.mean(c) for c in corners]) > 127: mask = cv2.bitwise_not(mask)
    
    cv2.rectangle(mask, (0, 0), (w, h), 0, 20)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return np.ones_like(gray) * 255
    
    c = max(contours, key=cv2.contourArea)
    pcb_mask = np.zeros_like(gray)
    cv2.drawContours(pcb_mask, [c], -1, 255, -1)
    return cv2.erode(pcb_mask, np.ones((5,5), np.uint8), iterations=2)

def segment_bright_objects(img_gray, pcb_mask):
    # 1. Top-Hat to highlight bright spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.TOPHAT_KERNEL_SIZE, cfg.TOPHAT_KERNEL_SIZE))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    
    # 2. Threshold
    _, solder_raw = cv2.threshold(tophat, cfg.SOLDER_THRESH_VAL, 255, cv2.THRESH_BINARY)
    solder_mask = cv2.bitwise_and(solder_raw, solder_raw, mask=pcb_mask)
    
    # 3. Trace Removal (kernel 9x9)
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.TRACE_REMOVAL_KERNEL, cfg.TRACE_REMOVAL_KERNEL))
    solder_clean = cv2.morphologyEx(solder_mask, cv2.MORPH_OPEN, clean_kernel)
    
    return solder_clean

def remove_scratches(mask):
    """Filter out long thin artifacts (scratches) using Aspect Ratio."""
    # Light opening to detach thin lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)
    
    for cnt in contours:
        if cv2.contourArea(cnt) < cfg.SOLDER_MIN_SIZE: continue
        
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        if w == 0 or h == 0: continue
            
        ar = max(w, h) / min(w, h)
        # AR > 3.5 is likely a scratch/line artifact
        if ar > 3.5: continue 
            
        cv2.drawContours(final_mask, [cnt], -1, 255, -1)
        
    return final_mask

def filter_and_subtract_silk(bright_mask, ref_silk_mask):
    # 1. Create Clean Seed (Solder Core)
    silk_heavy = cv2.dilate(ref_silk_mask, np.ones((5,5), np.uint8), iterations=cfg.MASK_DILATE_ITER)
    raw_seed = cv2.subtract(bright_mask, silk_heavy)
    
    # Clean noise in seed
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seed_opened = cv2.morphologyEx(raw_seed, cv2.MORPH_OPEN, clean_kernel, iterations=1)
    clean_seed = io.clean_noise(seed_opened, min_size=20)

    # 2. Create Barrier Mask (Bridge Cutting)
    silk_barrier = cv2.dilate(ref_silk_mask, np.ones((5,5), np.uint8), iterations=2) 
    growth_mask = cv2.subtract(bright_mask, silk_barrier)
    growth_mask = cv2.bitwise_or(growth_mask, clean_seed)

    # 3. Morphological Reconstruction
    final_float = reconstruction(clean_seed, growth_mask, method='dilation')
    final_mask = final_float.astype(np.uint8)

    # 4. Remove Scratches (Final Polish)
    final_clean = remove_scratches(final_mask)

    # 5. Extract Contours & Data
    contours, _ = cv2.findContours(final_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Size Filter
        if area < cfg.SOLDER_MIN_SIZE or area > cfg.SOLDER_MAX_SIZE: continue
        
        # Geometry Filter
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w)/h if h > 0 else 0
        if aspect < 1: aspect = 1/aspect
        if aspect > cfg.SOLDER_MAX_ASPECT_RATIO: continue

        # Convex Hull & Solidity Check
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0 and (float(area)/hull_area < cfg.SOLDER_MIN_SOLIDITY):
            continue

        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
        
        results.append({'contour': cnt, 'area': area, 'center': (cX, cY)})
            
    return final_clean, results

def analyze_solder_blobs(solder_mask):
    return [], None