import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import cv2

# --- 1. Параметры модели ---
N_RANGE = 5
IMAGE_SIZE = 512
K_TO_PIXEL_FACTOR = 60.0
WINDOW_STD_DEV = 0.65
B_FACTOR = 0.01
INTENSITY_THRESHOLD = 1e-8

# --- 2. Канонические базисные векторы ---
j = np.arange(5)
angles_par = 2.0 * np.pi * j / 5.0
angles_perp = 4.0 * np.pi * j / 5.0

kx_basis = np.sqrt(2.0/5.0) * np.cos(angles_par)
ky_basis = np.sqrt(2.0/5.0) * np.sin(angles_par)

kp_x_basis = np.sqrt(2.0/5.0) * np.cos(angles_perp)
kp_y_basis = np.sqrt(2.0/5.0) * np.sin(angles_perp)
kp_z_basis = np.full(5, np.sqrt(1.0/5.0))

def get_window_intensity(k_perp_sq):
    std_dev_sq = WINDOW_STD_DEV**2
    form_factor = np.exp(-k_perp_sq / (2.0 * std_dev_sq))
    return form_factor**2

# --- 3. Расчёт интенсивностей ---
image_detector = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
center_pixel = IMAGE_SIZE // 2
idx_range = np.arange(-N_RANGE, N_RANGE + 1)

for n0 in tqdm(idx_range, desc="Progress"):
    for n1 in idx_range:
        for n2 in idx_range:
            for n3 in idx_range:
                for n4 in idx_range:
                    n = np.array([n0, n1, n2, n3, n4])
                    kx = np.dot(n, kx_basis)
                    ky = np.dot(n, ky_basis)
                    kp_x = np.dot(n, kp_x_basis)
                    kp_y = np.dot(n, kp_y_basis)
                    kp_z = np.dot(n, kp_z_basis)

                    k_perp_norm_sq = kp_x**2 + kp_y**2 + kp_z**2
                    intensity = get_window_intensity(k_perp_norm_sq) * np.exp(-B_FACTOR * (kx**2 + ky**2))

                    if intensity > INTENSITY_THRESHOLD:
                        ix = int(round(kx * K_TO_PIXEL_FACTOR + center_pixel))
                        iy = int(round(ky * K_TO_PIXEL_FACTOR + center_pixel))
                        if 0 <= ix < IMAGE_SIZE and 0 <= iy < IMAGE_SIZE:
                            image_detector[iy, ix] += intensity

# --- 4. Логарифмическая нормализация ---
norm = LogNorm(vmin=max(image_detector.min(), 1e-10), vmax=image_detector.max())
normalized_image = norm(image_detector)
normalized_image = np.clip(normalized_image, 0, 1)  # от 0 до 1

# --- 5. Сохранение и отображение ---
output_filename = "quasicrystal_N5.png"
plt.imsave(output_filename, normalized_image, cmap='gray_r')
print(f"✅ Изображение сохранено как {output_filename}")

# --- 6. Открытие в системном окне (OpenCV) ---
img = cv2.imread(output_filename)
if img is not None:
    cv2.imshow("Квазикристалл (N=5)", img)
    print("🔍 Нажмите ESC для закрытия окна.")
    while True:
        if cv2.waitKey(1) == 27:  # ESC
            break
    cv2.destroyAllWindows()
else:
    print("⚠️ Ошибка: не удалось открыть файл изображения.")
