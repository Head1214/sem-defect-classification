import cv2 as cv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# Load SEM image
img = cv.imread("opencv-course-master/Resources/Photos/ChatGPT.png")
img = img[:-64, :]
output_img = img.copy()
gray_0 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray_0, (5, 5), 1)



# Apply Canny for sharp edge detection
edges = cv.Canny(gray, 125, 175)

# Create binary mask using Otsu thresholding and clean small noise
_, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
binary = cv.morphologyEx(binary, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

# Compute distance transform and suppress distance near edges
dist_transform = cv.distanceTransform(binary, cv.DIST_L2, 5)
dist_transform[edges > 0] = 0

# Detect local maxima as markers
local_max = peak_local_max(dist_transform, min_distance=3, labels=binary)
local_max_mask = np.zeros_like(dist_transform, dtype=bool)
local_max_mask[tuple(local_max.T)] = True
markers = ndi.label(local_max_mask)[0]

# Apply watershed
labels = watershed(-dist_transform, markers, mask=binary)

# Convert to uint8 mask for contour extraction
labels_uint8 = np.uint8(labels > 0) 
contours, _ = cv.findContours(labels_uint8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(f"Contours after watershed segmentation: {len(contours)}")

# Pixel to micrometer ratio
pixel_to_micrometer = 0.0253
# pixel_to_micrometer = 0.2747
# pixel_to_micrometer = 0.1315
# pixel_to_micrometer = 0.9090
# pixel_to_micrometer = 0.537 500x

# Create directories
particle_dir = "ChatGPT_particle"
fine_dir = "ChatGPT_fine"
agg_dir = "ChatGPT_agg"
os.makedirs(particle_dir, exist_ok=True)
os.makedirs(fine_dir, exist_ok=True)
os.makedirs(agg_dir, exist_ok=True)

# Feature extraction containers
final_contours = []
data_particles, data_fines, data_agglomeration = [], [], []
final_contours, final_fine_contours, final_agglomeration_contours = [], [], []
seen_centroids = []
seen_features = set()

def compute_curvature(contour):
    if len(contour) < 5:
        return 0
    contour = contour[:, 0, :]
    dx_dt = np.gradient(contour[:, 0])
    dy_dt = np.gradient(contour[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    denominator = (dx_dt**2 + dy_dt**2)**(3/2)
    epsilon = 1e-6
    return np.abs(d2x_dt2 * dy_dt - d2y_dt2 * dx_dt).mean() / (denominator + epsilon).mean()

def count_local_maxima(mask):
    dist = cv.distanceTransform(mask, cv.DIST_L2, 5)
    peaks = peak_local_max(dist, min_distance=5, labels=mask)
    return len(peaks)

for i, cnt in enumerate(contours):
    x, y, w, h = cv.boundingRect(cnt)
    if x <= 5 or y <= 5 or (x + w) >= (img.shape[1] - 5) or (y + h) >= (img.shape[0] - 5):
        continue

    area = cv.contourArea(cnt) * (pixel_to_micrometer ** 2)
    perimeter = cv.arcLength(cnt, True) * pixel_to_micrometer
    aspect_ratio = w / h if h != 0 else 0
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    solidity = area / (w * h * (pixel_to_micrometer ** 2)) if (w * h) != 0 else 0
    equivalent_diameter = np.sqrt(4 * area / np.pi)
    _, (MA, ma), angle = cv.fitEllipse(cnt) if len(cnt) >= 5 else ((0, 0), (0, 0), 0)
    elongation = ma / MA if MA > 0 else 0
    curvature = compute_curvature(cnt)

    M = cv.moments(cnt)
    centroid_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
    centroid_y = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2

    centroid = (centroid_x, centroid_y)
    if any(np.linalg.norm(np.array(centroid) - np.array(c)) < 5 for c in seen_centroids):
        continue
    seen_centroids.append(centroid)

    feature_signature = (round(area, 2), round(perimeter, 2))
    if feature_signature in seen_features:
        continue
    seen_features.add(feature_signature)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv.drawContours(mask, [cnt], -1, 255, -1)
    local_max_count = count_local_maxima(mask)

    feature_crop = img[y:y+h, x:x+w].copy()
    row_data = [area, perimeter, aspect_ratio, circularity, solidity,
                equivalent_diameter, angle, curvature, elongation, local_max_count,
                centroid_x, centroid_y, x, y, w, h]


    if 0.3< area < 1.5 and circularity >= 0.7:
        feature_filename = os.path.join(particle_dir, f"particle_feature_{i}.png")
        data_particles.append(row_data + [feature_filename])
        cv.drawContours(feature_crop, [cnt - [x, y]], -1, (0, 0, 255), 1)
        cv.imwrite(feature_filename, feature_crop)
        final_contours.append(cnt)
    elif 0.00785 <= area <= 0.30 and circularity > 0.2:
        feature_filename = os.path.join(fine_dir, f"fine_feature_{i}.png")
        data_fines.append(row_data + [feature_filename])
        cv.drawContours(feature_crop, [cnt - [x, y]], -1, (0, 255, 0), 1)
        cv.imwrite(feature_filename, feature_crop)
        final_fine_contours.append(cnt)
    elif area > 3.14 and (equivalent_diameter > 9.8 or circularity < 0.75 or curvature > 2 or elongation > 1.6 or local_max_count > 1):
        feature_filename = os.path.join(agg_dir, f"Agglomeration_feature_{i}.png")
        data_agglomeration.append(row_data + [feature_filename])
        cv.drawContours(feature_crop, [cnt - [x, y]], -1, (255, 0, 0), 1)
        cv.imwrite(feature_filename, feature_crop)
        final_agglomeration_contours.append(cnt)
    
    # row_data = [area, perimeter, aspect_ratio, circularity, solidity,
    #             equivalent_diameter, angle, curvature, elongation, local_max_count,
    #             centroid_x, centroid_y, x, y, w, h, feature_filename]
    

# Save CSV
columns = ["Area (µm²)", "Perimeter (µm)", "Aspect Ratio", "Circularity",
           "Solidity", "Diameter (µm)", "Orientation (°)", "Curvature",
           "Elongation", "Local Max Count", "Centroid X", "Centroid Y", "X", "Y", "Width", "Height", "Feature Image"]
pd.DataFrame(data_particles, columns=columns).to_csv("ChatGPT_particle.csv", index=False)
pd.DataFrame(data_fines, columns=columns).to_csv("ChatGPT_fine.csv", index=False)
pd.DataFrame(data_agglomeration, columns=columns).to_csv("ChatGPT_agg.csv", index=False)

# Draw overlays
cv.drawContours(output_img, final_fine_contours, -1, (0, 0, 255), 2)
cv.drawContours(output_img, final_contours, -1, (0, 255, 0), 2)
cv.drawContours(output_img, final_agglomeration_contours, -1, (255, 0, 0), 2)
plt.figure(figsize=(10, 6))
plt.imshow(cv.cvtColor(output_img, cv.COLOR_BGR2RGB))
plt.title("ChatGPT_Overlayed Final Classification Contours")
plt.axis("off")
plt.show()

print(f"Extracted {len(data_particles)} particle features and saved images.")
print(f"Extracted {len(data_fines)} fine particle features and saved images.")
print(f"Extracted {len(data_agglomeration)} agglomeration particle features and saved images.")
print("CSV saved as 'ChatGPT_particle.csv'")
print("Fine particle CSV saved as 'ChatGPT_fine.csv'")
print("Agglomeration particle CSV saved as 'ChatGPT_agg.csv'")