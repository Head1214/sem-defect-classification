# Updated version of the script:
# - Removes the break bridge morphological step
# - Keeps convex hull wrapping for consistency
# - Uses particle_mask from the non-background mask directly (with basic clean-up only)

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image

# === CONFIG ===
input_image_path = "opencv-course-master/Resources/Photos/ChatGPT xs2.png"
pixel_to_micrometer = 0.00625
L2_radius_um = 1.2
K = 3  # K-means clusters

# === LOAD IMAGE ===
img = cv.imread(input_image_path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# === K-MEANS CLUSTERING ===
Z = gray.reshape((-1, 1))
Z = np.float32(Z)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label_kmeans, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label_kmeans.flatten()]
res2 = res.reshape(gray.shape)

# === SELECT NON-BACKGROUND ===
sorted_centers = np.sort(center.flatten())
background_value = sorted_centers[0]
particle_mask_init = np.uint8(res2 != background_value) * 255

# === BASIC CLEANUP (NO BREAK BRIDGES) ===
# kernel_basic = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
# particle_mask = cv.morphologyEx(particle_mask_init, cv.MORPH_CLOSE, kernel_basic, iterations=3)
# particle_mask = cv.morphologyEx(particle_mask, cv.MORPH_OPEN, kernel_basic, iterations=2)

# === LARGEST COMPONENT ONLY ===
num_labels, labels_im = cv.connectedComponents(particle_mask_init)
max_area = 0
max_label = 1
for label_idx in range(1, num_labels):
    area = np.sum(labels_im == label_idx)
    if area > max_area:
        max_area = area
        max_label = label_idx
final_mask_clean = np.uint8(labels_im == max_label) * 255

# === CONVEX HULL ===
binary_mask = final_mask_clean > 0
convex_hull = convex_hull_image(binary_mask)
final_particle_mask = np.uint8(convex_hull) * 255

# === OTSU PORE MASK ===
_, pore_mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# === DISTANCE MAP & POROSITY ===
M = cv.moments(np.uint8(final_particle_mask))
center_x = int(M["m10"] / M["m00"])
center_y = int(M["m01"] / M["m00"])
Y, X = np.ogrid[:gray.shape[0], :gray.shape[1]]
dist_map = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2) * pixel_to_micrometer

L2_mask = np.logical_and(dist_map <= L2_radius_um, final_particle_mask == 255)
L1_mask = np.logical_and(dist_map > L2_radius_um, final_particle_mask == 255)

particle_area_px = np.sum(final_particle_mask == 255)
particle_pore_px = np.sum(np.logical_and(pore_mask == 255, final_particle_mask == 255))
total_porosity_ratio = particle_pore_px / particle_area_px

L2_area_px = np.sum(L2_mask)
L2_pore_px = np.sum(np.logical_and(pore_mask == 255, L2_mask))
L2_porosity_ratio = L2_pore_px / L2_area_px

L1_area_px = np.sum(L1_mask)
L1_pore_px = np.sum(np.logical_and(pore_mask == 255, L1_mask))
L1_porosity_ratio = L1_pore_px / L1_area_px

# === VISUALIZATION ===
overlay_img = img.copy()
L2_radius_px = int(L2_radius_um / pixel_to_micrometer)
cv.circle(overlay_img, (center_x, center_y), L2_radius_px, (0, 255, 0), 2)
contours, _ = cv.findContours(final_particle_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(overlay_img, contours, -1, (255, 0, 0), 2)

plt.figure(figsize=(18, 6))
plt.subplot(1, 4, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(res2, cmap="jet")
plt.title("K-Means Result (K=3)")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(final_particle_mask, cmap="gray")
plt.title("Final Particle Mask (Convex Hull)")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(overlay_img[..., ::-1])
plt.title("Overlay with L2 Circle + Hull")
plt.axis("off")
plt.tight_layout()
plt.show()



# === POROSITY AREA IN µm² ===
pixel_area_um2 = pixel_to_micrometer ** 2
total_pore_area_um2 = particle_pore_px * pixel_area_um2
L2_pore_area_um2 = L2_pore_px * pixel_area_um2
L1_pore_area_um2 = L1_pore_px * pixel_area_um2

# === TERMINAL REPORT ===
print("==== UPDATED POROSITY ANALYSIS ====")
print(f"Total porosity ratio: {total_porosity_ratio * 100:.2f}%")
print(f"L2 (<= {L2_radius_um} µm) porosity ratio: {L2_porosity_ratio * 100:.2f}%")
print(f"L1 (> {L2_radius_um} µm) porosity ratio: {L1_porosity_ratio * 100:.2f}%")
print()
print("---- PORE AREA (µm²) ----")
print(f"Total pore area: {total_pore_area_um2:.3f} µm²")
print(f"L2 pore area:    {L2_pore_area_um2:.3f} µm²")
print(f"L1 pore area:    {L1_pore_area_um2:.3f} µm²")
# 
# # Report results
# print("==== UPDATED POROSITY ANALYSIS ====")
# print(f"Total porosity ratio: {total_porosity_ratio * 100:.2f}%")
# print(f"L2 (<= {L2_radius_um} µm) porosity ratio: {L2_porosity_ratio * 100:.2f}%")

# print(f"L1 (> {L2_radius_um} µm) porosity ratio: {L1_porosity_ratio * 100:.2f}%")

