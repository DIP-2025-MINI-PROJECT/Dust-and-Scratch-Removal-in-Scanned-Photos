# restore_scratches_improved.py
# Usage:
# python restore_scratches_improved.py --input input/scanned.jpg --output output/cleaned.jpg

import argparse
import os
import cv2
import numpy as np
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def median_denoise(gray, ksize=3):
    return cv2.medianBlur(gray, ksize)

def detect_dust_scratch_mask_improved(gray,
                                      median_ksize=3,
                                      morph_selem_size=7,
                                      thr=38,
                                      min_size=10,
                                      max_blob_area_ratio=0.02,
                                      keep_thinness_ratio=0.25,
                                      dilate_iters=0):
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray_med = cv2.medianBlur(gray, median_ksize)

    selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_selem_size, morph_selem_size))
    tophat = cv2.morphologyEx(gray_med, cv2.MORPH_TOPHAT, selem)
    blackhat = cv2.morphologyEx(gray_med, cv2.MORPH_BLACKHAT, selem)
    combined = cv2.add(tophat, blackhat)
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)

    _, mask = cv2.threshold(combined, thr, 255, cv2.THRESH_BINARY)

    ker_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker_small, iterations=1)

    mask_bool = mask.astype(bool)
    mask_clean = remove_small_objects(mask_bool, min_size=min_size).astype(np.uint8) * 255

    lbl = label(mask_clean)
    props = regionprops(lbl)
    h, w = gray.shape
    max_blob_area = max_blob_area_ratio * (h*w)

    keep_mask = np.zeros_like(mask_clean)
    for p in props:
        area = p.area
        if area < min_size:
            continue
        if area > max_blob_area:
            continue
        try:
            minor = p.minor_axis_length
            major = p.major_axis_length
            thinness = minor / (major + 1e-6) if major > 0 else 1.0
        except Exception:
            thinness = 1.0
        if (area <= 5000) or (thinness < keep_thinness_ratio):
            keep_mask[lbl == p.label] = 255

    if dilate_iters > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        keep_mask = cv2.dilate(keep_mask, ker, iterations=dilate_iters)

    keep_mask = cv2.medianBlur(keep_mask, 3)
    return keep_mask

def inpaint_multi_pass(img, mask, method='telea', radii=(1,2)):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    res = img.copy()
    flags = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
    working_mask = mask.copy()
    for r in radii:
        res = cv2.inpaint(res, working_mask, int(r), flags)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        working_mask = cv2.erode(working_mask, ker, iterations=1)
    return res

def show_images(images, titles=None, figsize=(12,6)):
    n = len(images)
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(1, n, i+1)
        if img.ndim == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def process_file(input_path, output_path, params, show=False):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    den = median_denoise(gray, ksize=params['median_ksize'])

    mask = detect_dust_scratch_mask_improved(den,
                                             median_ksize=params['mask_median_ksize'],
                                             morph_selem_size=params['morph_selem_size'],
                                             thr=params['mask_threshold'],
                                             min_size=params['mask_min_size'],
                                             max_blob_area_ratio=params['mask_max_blob_area_ratio'],
                                             keep_thinness_ratio=params['mask_keep_thinness_ratio'],
                                             dilate_iters=params['dilate_iters'])

    result = inpaint_multi_pass(img, mask, method=params['inpaint_method'], radii=params['inpaint_radii'])

    base, ext = os.path.splitext(output_path)
    mask_path = base + "_mask.png"
    cv2.imwrite(output_path, result)
    cv2.imwrite(mask_path, mask)
    print(f"Saved: {output_path}")
    print(f"Saved: {mask_path}")

    if show:
        show_images([img, mask, result], titles=["Original", "Mask", "Result"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    params = {
        'median_ksize': 3,
    	'mask_median_ksize': 3,

    	'morph_selem_size': 9,       # slightly smaller
    	'mask_threshold': 46,        # higher = less sensitive noise

       	'mask_min_size': 120,        # increased from 10 → 120

       	'mask_max_blob_area_ratio': 0.015,

    	'mask_keep_thinness_ratio': 0.25,
    	'dilate_iters': 2,
    	'inpaint_method': 'ns',       # smoother for cracks
    	'inpaint_radii': (3, 4, 5)   
     }

    process_file(args.input, args.output, params, show=args.show)
