import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

# --- 1. Параметры модели ---
N_RANGE = 5
IMAGE_SIZE = 512
K_TO_PIXEL_FACTOR = 60.0
WINDOW_STD_DEV = 0.65
B_FACTOR = 0.01
INTENSITY_THRESHOLD = 1e-8

# --- 2. Канонические базисные векторы ---
print("Создание канонических матриц проекции...") # Added print statement back
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

# --- 3. Расчет ---
print("Запуск расчета 2D Фурье-образа (метод 2D-детектора)...") # Added print statement back
print(f"Диапазон индексов: [{-N_RANGE}, {N_RANGE}]") # Added print statement back
print(f"Размер детектора: {IMAGE_SIZE}x{IMAGE_SIZE}") # Added print statement back

image_detector = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
center_pixel = IMAGE_SIZE // 2
idx_range = np.arange(-N_RANGE, N_RANGE + 1)
total_peaks_added = 0

for n0 in tqdm(idx_range, desc="Progress"):
    for n1 in idx_range:
        for n2 in idx_range:
            for n3 in idx_range:
                for n4 in idx_range:
                    n_vector = np.array([n0, n1, n2, n3, n4])

                    kx = np.dot(n_vector, kx_basis)
                    ky = np.dot(n_vector, ky_basis)

                    kp_x = np.dot(n_vector, kp_x_basis)
                    kp_y = np.dot(n_vector, kp_y_basis)
                    kp_z = np.dot(n_vector, kp_z_basis)

                    k_perp_norm_sq = kp_x**2 + kp_y**2 + kp_z**2
                    intensity_window = get_window_intensity(k_perp_norm_sq)

                    k_vec_norm_sq = kx**2 + ky**2
                    intensity_atomic = np.exp(-B_FACTOR * k_vec_norm_sq)

                    total_intensity = intensity_window * intensity_atomic

                    if total_intensity > INTENSITY_THRESHOLD:
                        kx_px = kx * K_TO_PIXEL_FACTOR + center_pixel
                        ky_px = ky * K_TO_PIXEL_FACTOR + center_pixel

                        ix = int(round(kx_px))
                        iy = int(round(ky_px))

                        if 0 <= ix < IMAGE_SIZE and 0 <= iy < IMAGE_SIZE:
                            image_detector[iy, ix] += total_intensity
                            total_peaks_added += 1

print(f"Расчет завершен. Добавлено {total_peaks_added} вкладов в пиксели.") # Added print statement back

# --- 4. Визуализация ---
print("Отрисовка и сохранение дифракционной картины...") # Added print statement back

if total_peaks_added > 0:
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.set_facecolor('black')
    min_val_display = np.max(image_detector) * 1e-6

    # --- Ensure vmax is greater than vmin ---
    vmax_display = np.max(image_detector)
    if vmax_display <= min_val_display:
        print(f"Warning: Max intensity ({vmax_display:.2e}) <= vmin ({min_val_display:.2e}). Adjusting.")
        min_val_display = max(vmax_display * 0.01, 1e-12) # Use 1% of max or a tiny floor value

    ax.imshow(
        image_detector,
        cmap='gray_r',
        norm=LogNorm(vmin=max(min_val_display, 1e-10), vmax=vmax_display), # Keep floor value protection
        interpolation='nearest'
    )
    ax.set_title("SAED 2D-квазикристалл (5D-метод, N=5)", color='white')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    # --- SAVE FIGURE ---
    output_filename = "saed_simulation_N5_Nearest.png"
    plt.savefig(output_filename, dpi=300, facecolor='black', bbox_inches='tight')
    print(f"Изображение сохранено в файл: {output_filename}") # Confirmation message
    # --- END SAVE ---

    plt.show()
else:
    print("Пики не найдены.")