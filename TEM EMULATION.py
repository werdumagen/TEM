import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

# --- 1. Загрузка и подготовка изображения ---
image_path = 'input_image.png'

if not os.path.exists(image_path):
    print(f"Ошибка: Файл '{image_path}' не найден.")
    print("Пожалуйста, убедитесь, что файл находится в той же папке, что и скрипт.")
else:
    image_raw = imread(image_path)

    if image_raw.ndim == 3:
        image_gray = image_raw[:, :, 0]
    else:
        image_gray = image_raw

    # Инвертируем, чтобы фон был ~0, а атомы > 0
    image_structure = np.max(image_gray) - image_gray

    # --- 2. Создание "Суперячейки" (Tiling) ---
    # Повторим исходную структуру 4x4 раза
    # Это устранит артефакт "эффекта формы" от резких краев
    tiled_structure = np.tile(image_structure, (4, 4))

    # --- 3. Вычисление дифракции (FFT) ---
    # Теперь вычисляем FFT от "замощенного" изображения
    fft_raw = np.fft.fft2(tiled_structure)
    fft_centered = np.fft.fftshift(fft_raw)

    # Интенсивность = |Амплитуда|^2
    intensity = np.abs(fft_centered) ** 2

    # Используем логарифмическую шкалу, чтобы увидеть слабые пики
    log_intensity = np.log1p(intensity)

    # --- 4. Визуализация ---
    plt.figure(figsize=(16, 8))

    # Исходная структура
    plt.subplot(1, 2, 1)
    plt.title('Исходная структура (Реальное пространство)')
    plt.imshow(image_gray, cmap='gray')
    plt.axis('off')

    # Симуляция SAED
    plt.subplot(1, 2, 2)
    plt.title('Корректная симуляция SAED (Обратное пространство)')

    # "Обрежем" края, чтобы лучше было видно центр
    # (FFT от tiled-структуры очень четкое, но нам интересна симметрия в центре)
    # Определим, какую часть изображения показать
    h, w = log_intensity.shape
    crop_h, crop_w = h // 4, w // 4  # Показываем центральную 1/4 часть
    center_h, center_w = h // 2, w // 2

    cropped_log_intensity = log_intensity[center_h - crop_h: center_h + crop_h,
    center_w - crop_w: center_w + crop_w]

    plt.imshow(cropped_log_intensity, cmap='hot', aspect='auto')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('corrected_saed_simulation.png')
    print("Изображение с корректной симуляцией SAED сохранено как 'corrected_saed_simulation.png'")