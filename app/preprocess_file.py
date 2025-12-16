import cv2
import numpy as np
from skimage import color, filters, morphology
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.util import img_as_ubyte


def load_image(path, size=256):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    
    return canvas

def dullrazor(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    img_inpaint = cv2.inpaint(img_rgb, mask, 3, cv2.INPAINT_TELEA)
    return img_inpaint

def denoise(img_rgb):
    return cv2.fastNlMeansDenoisingColored(img_rgb, None, 3, 3, 7, 21)
    
def normalize_color(img_rgb):
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img_lab = cv2.merge([l, a, b])
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

def segment_otsu(img_rgb):
    b_channel = img_rgb[:, :, 2]
    h, w = b_channel.shape
    center = (w // 2, h // 2)
    radius = int(min(h, w) * 0.8 / 2) # Keep inner 90%
    circular_mask = np.zeros_like(b_channel, dtype=np.uint8)
    cv2.circle(circular_mask, center, radius, 255, -1)
    b_channel_masked = b_channel.copy()
    b_channel_masked[circular_mask == 0] = 255
    thresh = filters.threshold_otsu(b_channel_masked)
    mask = (b_channel_masked < thresh).astype(np.uint8)
    mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=500).astype(np.uint8)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=500).astype(np.uint8)
    return mask

def extract_features(img_rgb, mask):
    features = []

    gray = img_as_ubyte(color.rgb2gray(img_rgb))
    glcm = graycomatrix(gray, distances=[1, 3, 5], angles=[0,np.pi/4,np.pi/2], symmetric=True, normed=True)
    for prop in ['contrast', 'energy', 'homogeneity', 'correlation']:
        features.append(graycoprops(glcm, prop).mean())

    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    features.append(lbp.mean())

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    features += list(hsv.mean(axis=(0,1)))
    features += list(img_rgb.mean(axis=(0,1)))
    features += list(img_rgb.std(axis=(0,1)))

    if mask is not None:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            circularity = 4*np.pi*area/(perimeter**2) if perimeter>0 else 0
            features.append(circularity)
            M = cv2.moments(c)
            eccentricity = ((M['mu20'] - M['mu02'])**2 - 4*(M['mu11']**2))/((M['mu20']+M['mu02'])**2) if (M['mu20']+M['mu02'])!=0 else 0
            features.append(float(eccentricity))
            hull = cv2.convexHull(c)
            solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull)>0 else 0
            features.append(solidity)
        else:
            features += [0,0,0]

    return features