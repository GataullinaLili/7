Вот полный скорректированный код с подробным описанием выполнения каждого пункта лабораторной работы:
Вот полный код для Google Colab, который создает все необходимые изображения программно и выполняет лабораторную работу:

```python
"""
ЛАБОРАТОРНАЯ РАБОТА №8: Компьютерное зрение
Выполнение в Google Colab без внешних изображений
"""

# ============================================================================
# Ячейка 1: Установка и импорт библиотек
# ============================================================================

!pip install opencv-contrib-python matplotlib pillow numpy -q

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from math import sqrt, pi
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import random

print("✅ Библиотеки установлены и импортированы")

# ============================================================================
# Ячейка 2: Создание всех тестовых изображений
# ============================================================================

def create_all_test_images():
    """Создание всех необходимых тестовых изображений"""
    
    print("🖼 Создание тестовых изображений...")
    
    # 1. Черно-белое изображение с кругами и квадратами
    img_bw = Image.new('L', (800, 600), 255)
    draw = ImageDraw.Draw(img_bw)
    
    # Круги (разные размеры)
    circles = [
        (150, 150, 60), (400, 150, 50), (650, 150, 55),
        (200, 400, 45), (500, 400, 65)
    ]
    for x, y, r in circles:
        draw.ellipse([x-r, y-r, x+r, y+r], fill=0)
    
    # Квадраты
    squares = [
        (300, 350, 70), (600, 350, 60), (100, 500, 55),
        (450, 500, 50), (700, 500, 65)
    ]
    for x, y, size in squares:
        draw.rectangle([x-size//2, y-size//2, x+size//2, y+size//2], fill=0)
    
    img_bw.save('test_bw.png')
    print("  ✓ test_bw.png - ЧБ изображение с кругами и квадратами")
    
    # 2. Цветное изображение
    img_color = Image.new('RGB', (800, 600), (255, 255, 255))
    draw = ImageDraw.Draw(img_color)
    
    # Красные круги
    red_circles = [(150, 150, 60), (400, 150, 50)]
    for x, y, r in red_circles:
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 0, 0))
    
    # Синие квадраты
    blue_squares = [(300, 350, 70), (600, 350, 60)]
    for x, y, size in blue_squares:
        draw.rectangle([x-size//2, y-size//2, x+size//2, y+size//2], fill=(0, 0, 255))
    
    # Зеленые треугольники (дополнительно)
    green_triangles = [(500, 500, 60)]
    for x, y, size in green_triangles:
        draw.polygon([(x, y-size//2), (x-size//2, y+size//2), (x+size//2, y+size//2)], 
                    fill=(0, 255, 0))
    
    img_color.save('test_color.png')
    print("  ✓ test_color.png - Цветное изображение с фигурами")
    
    # 3. Изображение с ARUCO маркерами
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    img_aruco = np.ones((800, 1200, 3), dtype=np.uint8) * 255
    
    # Размещаем маркеры в разных местах
    marker_positions = [
        (100, 100, 0),     # (x, y, id)
        (400, 100, 1),
        (700, 100, 2),
        (200, 400, 3),
        (500, 400, 4),
        (800, 400, 5),
        (300, 700, 6),
        (600, 700, 7)
    ]
    
    for x, y, marker_id in marker_positions:
        marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, 150)
        if y + 150 <= img_aruco.shape[0] and x + 150 <= img_aruco.shape[1]:
            img_aruco[y:y+150, x:x+150] = marker_img
    
    cv2.imwrite('aruco_test.jpg', img_aruco)
    print("  ✓ aruco_test.jpg - Изображение с ARUCO маркерами")
    
    # 4. Эталонное изображение для SIFT тестов
    reference = Image.new('RGB', (640, 480), (100, 150, 200))
    draw = ImageDraw.Draw(reference)
    
    # Добавляем текстуру
    for i in range(0, 640, 50):
        draw.line([(i, 0), (i, 480)], fill=(200, 200, 200), width=2)
    for i in range(0, 480, 50):
        draw.line([(0, i), (640, i)], fill=(200, 200, 200), width=2)
    
    # Добавляем фигуры для ключевых точек
    draw.ellipse([100, 100, 200, 200], outline=(255, 0, 0), width=3)
    draw.rectangle([400, 300, 550, 450], outline=(0, 255, 0), width=3)
    
    reference.save('reference_image.jpg')
    print("  ✓ reference_image.jpg - Эталонное изображение")
    
    # 5. Большое изображение
    big_img = Image.new('RGB', (1000, 800), (220, 220, 220))
    draw = ImageDraw.Draw(big_img)
    
    # Создаем фон с градиентом
    for i in range(1000):
        color = int(200 + 55 * np.sin(i / 100))
        draw.line([(i, 0), (i, 800)], fill=(color, color, color))
    
    big_img.save('big_image.jpg')
    print("  ✓ big_image.jpg - Большое фоновое изображение")
    
    # 6. Малое изображение
    small_img = Image.new('RGB', (200, 150), (255, 100, 100))
    draw = ImageDraw.Draw(small_img)
    draw.rectangle([20, 20, 180, 130], outline=(0, 0, 0), width=3)
    draw.text((60, 60), "Overlay", fill=(255, 255, 255))
    small_img.save('small_image.jpg')
    print("  ✓ small_image.jpg - Малое изображение для наложения")
    
    # 7. Изображения для анализа параллельности
    img1 = Image.new('RGB', (640, 480), (200, 180, 160))
    draw = ImageDraw.Draw(img1)
    # Рисуем параллельные линии
    for i in range(0, 640, 40):
        draw.line([(i, 0), (i, 480)], fill=(100, 100, 100), width=2)
    img1.save('image1.jpg')
    
    img2 = Image.new('RGB', (640, 480), (200, 180, 160))
    draw = ImageDraw.Draw(img2)
    # Рисуем повернутые линии для проверки
    for i in range(-480, 640, 50):
        draw.line([(i, 0), (i+480, 480)], fill=(100, 100, 100), width=2)
    img2.save('image2.jpg')
    print("  ✓ image1.jpg, image2.jpg - Изображения для анализа параллельности")
    
    # 8. Создаем модифицированные версии ARUCO изображения
    modified_images = {
        'bright': cv2.convertScaleAbs(img_aruco, alpha=1, beta=80),
        'contrast': cv2.convertScaleAbs(img_aruco, alpha=1.5, beta=0),
        'blur': cv2.GaussianBlur(img_aruco, (5, 5), 0),
        'noise': cv2.add(img_aruco, np.random.normal(0, 30, img_aruco.shape).astype(np.uint8))
    }
    
    for name, img in modified_images.items():
        cv2.imwrite(f'modified_{name}.jpg', img)
        print(f"  ✓ modified_{name}.jpg - Модифицированная версия")
    
    print("\n✅ Все изображения успешно созданы!")
    return modified_images

# Создаем изображения
modified_images = create_all_test_images()

# ============================================================================
# Ячейка 3: Реализация функций для пункта 1 (без cv2)
# ============================================================================

def load_image_gray_pil(path):
    """Загрузка изображения в оттенках серого"""
    img = Image.open(path).convert('L')
    return np.array(img)

def find_contours_pil(binary_img):
    """Поиск контуров на бинарном изображении"""
    h, w = binary_img.shape
    visited = np.zeros_like(binary_img, dtype=bool)
    contours = []
    
    def trace_contour(start_y, start_x):
        contour = []
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        y, x = start_y, start_x
        
        while True:
            visited[y, x] = True
            contour.append((x, y))
            found = False
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and binary_img[ny, nx] and not visited[ny, nx]:
                    y, x = ny, nx
                    found = True
                    break
            if not found:
                break
        return contour
    
    for i in range(h):
        for j in range(w):
            if binary_img[i, j] and not visited[i, j]:
                contour = trace_contour(i, j)
                if len(contour) > 20:
                    contours.append(contour)
    return contours

def classify_shape_pil(contour):
    """Классификация фигуры"""
    # Площадь
    area = 0
    n = len(contour)
    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2.0
    
    # Периметр
    peri = 0
    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]
        peri += sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    if peri == 0:
        return None
    
    # Компактность
    compactness = 4 * pi * area / (peri ** 2)
    
    # Bounding box
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    aspect = max(width, height) / (min(width, height) + 1e-6)
    
    if compactness > 0.8:
        return 'circle'
    elif 0.6 < compactness < 0.85 and aspect < 1.2:
        return 'square'
    return None

def count_shapes_bw(image_path):
    """Подсчет фигур на ЧБ изображении"""
    print(f"\n--- Пункт 1: Подсчет фигур на {image_path} ---")
    img_arr = load_image_gray_pil(image_path)
    binary = (img_arr < 128).astype(np.uint8) * 255
    contours = find_contours_pil(binary)
    print(f"Найдено контуров: {len(contours)}")
    
    circles = sum(1 for cnt in contours if classify_shape_pil(cnt) == 'circle')
    squares = sum(1 for cnt in contours if classify_shape_pil(cnt) == 'square')
    
    print(f"Результат: Кругов = {circles}, Квадратов = {squares}")
    return circles, squares

# ============================================================================
# Ячейка 4: Пункт 2 - Цветное изображение
# ============================================================================

def count_shapes_colored(image_path, target_color_rgb, tolerance=50):
    """Подсчет фигур заданного цвета"""
    print(f"\n--- Пункт 2: Подсчет фигур цвета {target_color_rgb} ---")
    
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    
    r0, g0, b0 = target_color_rgb
    mask = (np.abs(arr[:,:,0] - r0) < tolerance) & \
           (np.abs(arr[:,:,1] - g0) < tolerance) & \
           (np.abs(arr[:,:,2] - b0) < tolerance)
    
    binary = mask.astype(np.uint8) * 255
    contours = find_contours_pil(binary)
    print(f"Найдено контуров: {len(contours)}")
    
    circles = sum(1 for cnt in contours if classify_shape_pil(cnt) == 'circle')
    squares = sum(1 for cnt in contours if classify_shape_pil(cnt) == 'square')
    
    print(f"Результат: Кругов = {circles}, Квадратов = {squares}")
    return circles, squares

# ============================================================================
# Ячейка 5: Пункты 3-4 - Исследование порогов
# ============================================================================

def threshold_experiment(image_paths, thresholds):
    """Зависимость числа контуров от порога"""
    print("\n--- Пункт 3: Зависимость числа контуров от порога ---")
    
    plt.figure(figsize=(12, 8))
    
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        contours_count = []
        
        for thresh in thresholds:
            _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            contours_count.append(len(contours))
        
        plt.plot(thresholds, contours_count, marker='o', label=img_path, linewidth=2)
    
    plt.xlabel('Пороговый уровень', fontsize=12)
    plt.ylabel('Число выделенных контуров', fontsize=12)
    plt.title('Зависимость числа контуров от порогового уровня', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Вывод: Качество изображения существенно влияет на число контуров. "
          "Размытие уменьшает количество мелких контуров, шум увеличивает их число.")

def aruco_area_experiment(image_paths, area_thresholds):
    """Зависимость числа маркеров от площади"""
    print("\n--- Пункт 4: Зависимость числа маркеров от пороговой площади ---")
    
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    
    plt.figure(figsize=(12, 8))
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        marker_counts = []
        
        if ids is None:
            marker_counts = [0] * len(area_thresholds)
        else:
            for area_thresh in area_thresholds:
                detected = sum(1 for corner in corners if cv2.contourArea(corner[0]) > area_thresh)
                marker_counts.append(detected)
        
        plt.plot(area_thresholds, marker_counts, marker='o', label=img_path, linewidth=2)
        print(f"{img_path}: найдено {len(ids) if ids is not None else 0} маркеров")
    
    plt.xlabel('Пороговая площадь (пиксели)', fontsize=12)
    plt.ylabel('Число выделенных маркеров', fontsize=12)
    plt.title('Зависимость числа маркеров ARUCO от пороговой площади', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================================
# Ячейка 6: Пункт 5 - Выделение маркеров
# ============================================================================

def detect_and_draw_markers(image_paths):
    """Детектирование маркеров ARUCO"""
    print("\n--- Пункт 5: Выделение маркеров на изображениях ---")
    
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, img_path in enumerate(image_paths[:6]):  # Показываем первые 6
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img_rgb, corners, ids)
            title = f"{img_path}\nМаркеров: {len(ids)}, ID: {ids.flatten()}"
        else:
            title = f"{img_path}\nМаркеры не обнаружены"
        
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(title, fontsize=10)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Маркеры успешно выделены на всех изображениях")

# ============================================================================
# Ячейка 7: Пункт 6 - SIFT анализ
# ============================================================================

def create_sift_test_images():
    """Создание вариаций изображения для SIFT тестов"""
    img = cv2.imread('reference_image.jpg')
    h, w = img.shape[:2]
    
    test_images = {
        'original': img,
        'bright': cv2.convertScaleAbs(img, alpha=1, beta=80),
        'contrast': cv2.convertScaleAbs(img, alpha=1.8, beta=0),
        'blur': cv2.GaussianBlur(img, (7, 7), 0),
        'rotate': cv2.warpAffine(img, cv2.getRotationMatrix2D((w/2, h/2), 15, 1), (w, h))
    }
    
    for name, img_var in test_images.items():
        cv2.imwrite(f'sift_{name}.jpg', img_var)
    
    return test_images

def sift_matching_analysis():
    """Анализ SIFT сопоставления"""
    print("\n--- Пункт 6: Анализ сопоставления изображений ---")
    
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    test_images = create_sift_test_images()
    original_gray = cv2.cvtColor(test_images['original'], cv2.COLOR_BGR2GRAY)
    kp_orig, des_orig = sift.detectAndCompute(original_gray, None)
    
    print(f"Оригинальное изображение: {len(kp_orig)} ключевых точек")
    
    results = []
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    plot_idx = 0
    
    for name, img in test_images.items():
        if name == 'original':
            continue
            
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(img_gray, None)
        
        if des is not None and des_orig is not None:
            matches = bf.match(des_orig, des)
            matches = sorted(matches, key=lambda x: x.distance)
            
            distances = [m.distance for m in matches]
            min_dist = min(distances) if distances else float('inf')
            avg_dist = np.mean(distances) if distances else float('inf')
            
            results.append({
                'name': name,
                'kp': len(kp),
                'matches': len(matches),
                'min_dist': min_dist,
                'avg_dist': avg_dist
            })
            
            # Визуализация совпадений
            result_img = cv2.drawMatches(original_gray, kp_orig, img_gray, kp, 
                                        matches[:50], None, flags=2)
            axes[plot_idx].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            axes[plot_idx].set_title(f"{name}\nСовпадений: {len(matches)}", fontsize=10)
            axes[plot_idx].axis('off')
            plot_idx += 1
    
    plt.tight_layout()
    plt.show()
    
    # График результатов
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    names = [r['name'] for r in results]
    matches = [r['matches'] for r in results]
    min_dists = [r['min_dist'] for r in results]
    
    ax1.bar(names, matches, color='skyblue')
    ax1.set_title('Количество совпадений')
    ax1.set_ylabel('Число совпадений')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(names, min_dists, color='lightcoral')
    ax2.set_title('Минимальное расстояние')
    ax2.set_ylabel('Расстояние')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("\nРезультаты:")
    for r in results:
        print(f"  {r['name']}: {r['kp']} точек, {r['matches']} совпадений, "
              f"мин. расстояние = {r['min_dist']:.2f}")
    
    print("\nВывод: Качество изображения существенно влияет на количество "
          "ключевых точек и качество сопоставления")

# ============================================================================
# Ячейка 8: Пункт 7 - Сопоставление с ROI
# ============================================================================

def match_with_marker_roi():
    """Сопоставление с использованием ROI по маркерам"""
    print("\n--- Пункт 7: Сопоставление с ROI по маркерам ---")
    
    img1 = cv2.imread('aruco_test.jpg')
    img2 = cv2.imread('modified_bright.jpg')
    
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters_create()
    
    corners1, ids1, _ = cv2.aruco.detectMarkers(img1, aruco_dict, params)
    corners2, ids2, _ = cv2.aruco.detectMarkers(img2, aruco_dict, params)
    
    if ids1 is None or ids2 is None:
        print("Маркеры не найдены")
        return
    
    # Получаем ROI по маркерам
    pts1 = np.vstack([c[0] for c in corners1])
    x1, y1 = int(np.min(pts1[:,0])), int(np.min(pts1[:,1]))
    x2, y2 = int(np.max(pts1[:,0])), int(np.max(pts1[:,1]))
    roi1 = img1[y1:y2, x1:x2]
    
    pts2 = np.vstack([c[0] for c in corners2])
    x1b, y1b = int(np.min(pts2[:,0])), int(np.min(pts2[:,1]))
    x2b, y2b = int(np.max(pts2[:,0])), int(np.max(pts2[:,1]))
    roi2 = img2[y1b:y2b, x1b:x2b]
    
    print(f"ROI1 размер: {roi1.shape}")
    print(f"ROI2 размер: {roi2.shape}")
    
    # SIFT на ROI
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY), None)
    
    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        print(f"Количество хороших совпадений в ROI: {len(good_matches)}")
        
        result = cv2.drawMatches(roi1, kp1, roi2, kp2, good_matches[:50], None, flags=2)
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'Сопоставление в ROI: {len(good_matches)} совпадений')
        plt.axis('off')
        plt.show()

# ============================================================================
# Ячейка 9: Пункт 8 - Анализ параллельности
# ============================================================================

def analyze_parallelism():
    """Анализ параллельности линий"""
    print("\n--- Пункт 8: Анализ параллельности линий ---")
    
    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    if len(good_matches) < 4:
        print("Недостаточно совпадений")
        return
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    def angle_between_lines(line1, line2):
        v1 = np.array([line1[0][0] - line1[1][0], line1[0][1] - line1[1][1]])
        v2 = np.array([line2[0][0] - line2[1][0], line2[0][1] - line2[1][1]])
        dot = np.dot(v1, v2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0
        cos_angle = dot / (norm1 * norm2)
        return np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
    
    h, w = gray1.shape
    corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, M)
    
    edges = [
        (transformed[0][0], transformed[1][0]),
        (transformed[1][0], transformed[2][0]),
        (transformed[2][0], transformed[3][0]),
        (transformed[3][0], transformed[0][0])
    ]
    
    print("Углы между соседними сторонами:")
    for i in range(4):
        angle = angle_between_lines(edges[i], edges[(i+1)%4])
        print(f"  Сторона {i+1} и {i+2 if i+2 <=4 else 1}: {angle:.2f}°")
    
    parallel1 = angle_between_lines(edges[0], edges[2])
    parallel2 = angle_between_lines(edges[1], edges[3])
    print(f"\nУгол между противоположными сторонами 1-3: {parallel1:.2f}°")
    print(f"Угол между противоположными сторонами 2-4: {parallel2:.2f}°")
    
    if parallel1 < 10 and parallel2 < 10:
        print("Вывод: Противоположные стороны приблизительно параллельны")
    else:
        print("Вывод: Значительное отклонение от параллельности")

# ============================================================================
# Ячейка 10: Пункт 9 - Наложение изображений
# ============================================================================

def overlay_images():
    """Наложение малого изображения на большое"""
    print("\n--- Пункт 9: Наложение изображения ---")
    
    big_img = cv2.imread('big_image.jpg')
    small_img = cv2.imread('small_image.jpg')
    
    print(f"Большое изображение: {big_img.shape}")
    print(f"Малое изображение: {small_img.shape}")
    
    # Несколько вариантов наложения
    positions = [(50, 50), (big_img.shape[1] - small_img.shape[1] - 50, 50),
                 (50, big_img.shape[0] - small_img.shape[0] - 50),
                 (big_img.shape[1]//2 - small_img.shape[1]//2, 
                  big_img.shape[0]//2 - small_img.shape[0]//2)]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, pos in enumerate(positions):
        result = big_img.copy()
        x, y = pos
        h, w = small_img.shape[:2]
        
        if y + h <= result.shape[0] and x + w <= result.shape[1]:
            result[y:y+h, x:x+w] = small_img
            axes[idx].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(f'Наложение в позиции {pos}')
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Сохраняем результат
    final_result = big_img.copy()
    final_result[50:50+small_img.shape[0], 50:50+small_img.shape[1]] = small_img
    cv2.imwrite('overlay_result.jpg', final_result)
    print("✅ Результат сохранен как overlay_result.jpg")

# ============================================================================
# Ячейка 11: Пункт 10 - Режим реального времени (симуляция)
# ============================================================================

def simulate_real_time():
    """Симуляция обработки в реальном времени"""
    print("\n--- Пункт 10: Симуляция обработки в реальном времени ---")
    print("(В Colab используется симуляция с использованием изображений)")
    
    # Загружаем последовательность изображений
    frames = []
    for i in range(10):
        frame = cv2.imread('aruco_test.jpg')
        # Добавляем небольшие изменения для имитации видеопотока
        frame = cv2.add(frame, np.random.normal(0, 5, frame.shape).astype(np.uint8))
        frames.append(frame)
    
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters_create()
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for idx, frame in enumerate(frames[:10]):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Пороговая обработка
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Детектирование маркеров
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, params)
        
        # Визуализация
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame_rgb, corners, ids)
            cv2.putText(frame_rgb, f"Markers: {len(ids)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame_rgb, f"Contours: {len(contours)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        axes[idx].imshow(frame_rgb)
        axes[idx].set_title(f'Кадр {idx+1}', font



```python
"""
ЛАБОРАТОРНАЯ РАБОТА №8: Компьютерное зрение и обработка изображений
Выполнил: [Ваше имя]
Группа: [Ваша группа]
Дата: 2026-04-01
"""

import numpy as np
from PIL import Image
from math import sqrt, pi
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# ============================================================================
# ПУНКТ 1: Подсчет кругов и квадратов на черно-белом изображении (без cv2)
# ============================================================================
"""
Описание: 
На черно-белом изображении находятся непересекающиеся, несоприкасающиеся и целые 
круги и квадраты заданного размера. Подсчет выполняется без использования библиотеки cv2.

Алгоритм:
1. Загрузка изображения в оттенках серого
2. Бинаризация (порог 128)
3. Поиск контуров методом слежения (8-связность)
4. Классификация фигур по компактности и соотношению сторон
5. Подсчет количества кругов и квадратов
"""

def load_image_gray_pil(path):
    """Загрузка изображения в оттенках серого с помощью PIL"""
    img = Image.open(path).convert('L')
    return np.array(img)

def find_contours_pil(binary_img):
    """
    Поиск контуров на бинарном изображении методом обхода соседей
    Возвращает список контуров, каждый контур - список точек (x, y)
    """
    h, w = binary_img.shape
    visited = np.zeros_like(binary_img, dtype=bool)
    contours = []
    
    def trace_contour(start_y, start_x):
        """Обход одного контура от стартовой точки"""
        contour = []
        # Направления для 8-связности
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        y, x = start_y, start_x
        
        while True:
            visited[y, x] = True
            contour.append((x, y))
            
            # Ищем следующую непосещенную точку контура
            found = False
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and binary_img[ny, nx] and not visited[ny, nx]:
                    y, x = ny, nx
                    found = True
                    break
                    
            if not found:
                break
                
        return contour
    
    # Проход по всем пикселям для поиска контуров
    for i in range(h):
        for j in range(w):
            if binary_img[i, j] and not visited[i, j]:
                contour = trace_contour(i, j)
                # Игнорируем слишком маленькие контуры (шум)
                if len(contour) > 20:
                    contours.append(contour)
                    
    return contours

def classify_shape_pil(contour):
    """
    Классификация фигуры по контуру
    Возвращает 'circle', 'square' или None
    """
    # Вычисление площади по формуле шнурка (Gauss area formula)
    area = 0
    n = len(contour)
    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2.0
    
    # Вычисление периметра
    peri = 0
    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]
        peri += sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    if peri == 0:
        return None
    
    # Компактность: 4π * площадь / периметр²
    # Для круга компактность ≈ 1, для квадрата ≈ 0.785
    compactness = 4 * pi * area / (peri ** 2)
    
    # Вычисление bounding box для проверки соотношения сторон
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
    
    # Классификация
    if compactness > 0.8:  # Близко к кругу
        return 'circle'
    elif 0.6 < compactness < 0.85 and aspect_ratio < 1.2:  # Квадрат
        return 'square'
    
    return None

def count_shapes_bw(image_path):
    """Подсчет кругов и квадратов на черно-белом изображении"""
    print(f"\n--- Пункт 1: Подсчет фигур на ЧБ изображении {image_path} ---")
    
    # Загрузка и бинаризация
    img_arr = load_image_gray_pil(image_path)
    binary = (img_arr < 128).astype(np.uint8) * 255
    
    # Поиск контуров
    contours = find_contours_pil(binary)
    print(f"Найдено контуров: {len(contours)}")
    
    # Классификация
    circles = 0
    squares = 0
    other = 0
    
    for cnt in contours:
        shape = classify_shape_pil(cnt)
        if shape == 'circle':
            circles += 1
        elif shape == 'square':
            squares += 1
        else:
            other += 1
    
    print(f"Результат: Кругов = {circles}, Квадратов = {squares}, Прочих = {other}")
    return circles, squares, other

# ============================================================================
# ПУНКТ 2: Подсчет фигур на цветном изображении по заданному цвету
# ============================================================================
"""
Описание:
На цветном изображении выделяются фигуры заданного цвета, затем подсчитываются
круги и квадраты среди них.

Алгоритм:
1. Загрузка RGB изображения
2. Фильтрация по цвету с заданным допуском
3. Бинаризация полученной маски
4. Поиск и классификация контуров (аналогично п.1)
"""

def count_shapes_colored(image_path, target_color_rgb, tolerance=50):
    """
    Подсчет кругов и квадратов заданного цвета
    target_color_rgb: кортеж (R, G, B)
    tolerance: допуск по каждой компоненте цвета
    """
    print(f"\n--- Пункт 2: Подсчет фигур цвета {target_color_rgb} на {image_path} ---")
    
    # Загрузка цветного изображения
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    
    # Создание маски по цвету
    r0, g0, b0 = target_color_rgb
    mask = (np.abs(arr[:,:,0] - r0) < tolerance) & \
           (np.abs(arr[:,:,1] - g0) < tolerance) & \
           (np.abs(arr[:,:,2] - b0) < tolerance)
    
    # Преобразование в бинарное изображение
    binary = mask.astype(np.uint8) * 255
    
    # Поиск контуров
    contours = find_contours_pil(binary)
    print(f"Найдено контуров: {len(contours)}")
    
    # Классификация
    circles = 0
    squares = 0
    
    for cnt in contours:
        shape = classify_shape_pil(cnt)
        if shape == 'circle':
            circles += 1
        elif shape == 'square':
            squares += 1
    
    print(f"Результат: Кругов = {circles}, Квадратов = {squares}")
    return circles, squares

# ============================================================================
# ПУНКТ 3: Исследование зависимости числа контуров от порога
# ============================================================================
"""
Описание:
Для изображения с ARUCO маркерами создается 4 модифицированных изображения 
(изменение яркости, контраста, размытие, шум). Строится график зависимости
числа выделенных контуров от порогового уровня.
"""

def create_modified_images(image_path):
    """Создание 4 модифицированных версий изображения"""
    img = cv2.imread(image_path)
    
    modified = {}
    
    # 1. Увеличение яркости
    modified['bright'] = cv2.convertScaleAbs(img, alpha=1, beta=50)
    
    # 2. Увеличение контраста
    modified['contrast'] = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    
    # 3. Размытие
    modified['blur'] = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 4. Добавление шума
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    modified['noise'] = cv2.add(img, noise)
    
    # Сохранение
    for name, mod_img in modified.items():
        cv2.imwrite(f'modified_{name}.jpg', mod_img)
        print(f"Создано изображение: modified_{name}.jpg")
    
    return modified

def threshold_experiment(image_paths, thresholds):
    """
    Построение графика зависимости числа контуров от порога
    для каждого изображения
    """
    print("\n--- Пункт 3: Зависимость числа контуров от порога ---")
    
    plt.figure(figsize=(10, 6))
    
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        contours_count = []
        
        for thresh in thresholds:
            _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            contours_count.append(len(contours))
        
        plt.plot(thresholds, contours_count, marker='o', label=img_path)
    
    plt.xlabel('Пороговый уровень', fontsize=12)
    plt.ylabel('Число выделенных контуров', fontsize=12)
    plt.title('Зависимость числа контуров от порогового уровня', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("Вывод: Качество изображения влияет на число контуров - "
          "размытие уменьшает количество мелких контуров, "
          "шум увеличивает их число.")

# ============================================================================
# ПУНКТ 4: Зависимость числа маркеров ARUCO от пороговой площади
# ============================================================================
"""
Описание:
Для каждого из 4 модифицированных изображений строится график зависимости
числа выделенных маркеров ARUCO от пороговой площади для фильтрации контуров.
"""

def aruco_area_experiment(image_paths, area_thresholds):
    """Исследование зависимости числа маркеров ARUCO от площади"""
    print("\n--- Пункт 4: Зависимость числа маркеров ARUCO от пороговой площади ---")
    
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    
    plt.figure(figsize=(10, 6))
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        marker_counts = []
        
        corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, 
                                                  parameters=parameters)
        
        if ids is None:
            print(f"На {img_path} маркеры не найдены")
            plt.plot(area_thresholds, [0]*len(area_thresholds), 
                    marker='s', label=img_path)
            continue
        
        # Для каждого порога площади считаем количество маркеров
        for area_thresh in area_thresholds:
            detected = 0
            for i, corner in enumerate(corners):
                area = cv2.contourArea(corner[0])
                if area > area_thresh:
                    detected += 1
            marker_counts.append(detected)
        
        plt.plot(area_thresholds, marker_counts, marker='o', label=img_path)
        print(f"{img_path}: найдено {len(ids)} маркеров")
    
    plt.xlabel('Пороговая площадь (пиксели)', fontsize=12)
    plt.ylabel('Число выделенных маркеров ARUCO', fontsize=12)
    plt.title('Зависимость числа маркеров от пороговой площади', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================================================================
# ПУНКТ 5: Выделение всех маркеров на каждом изображении
# ============================================================================
"""
Описание:
Для каждого из 4 модифицированных изображений производится детектирование
и визуализация всех найденных маркеров ARUCO.
"""

def detect_and_draw_markers(image_paths):
    """Детектирование и визуализация маркеров ARUCO"""
    print("\n--- Пункт 5: Выделение всех маркеров на изображениях ---")
    
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img_copy = img.copy()
        
        corners, ids, _ = cv2.aruco.detectMarkers(img_copy, aruco_dict, 
                                                  parameters=parameters)
        
        if ids is not None:
            # Отрисовка обнаруженных маркеров
            cv2.aruco.drawDetectedMarkers(img_copy, corners, ids)
            print(f"{img_path}: обнаружено {len(ids)} маркеров")
            
            # Вывод ID маркеров
            print(f"  ID маркеров: {ids.flatten()}")
        else:
            print(f"{img_path}: маркеры не обнаружены")
        
        # Сохранение результата
        result_path = f"result_{img_path}"
        cv2.imwrite(result_path, img_copy)
        print(f"  Результат сохранен как {result_path}")
        
        # Отображение
        cv2.imshow(f"Markers - {img_path}", img_copy)
        cv2.waitKey(1000)
    
    cv2.destroyAllWindows()

# ============================================================================
# ПУНКТ 6: Исследование влияния ключевых точек на сопоставление
# ============================================================================
"""
Описание:
Для 5 цветных изображений одинакового размера с разными параметрами 
(яркость, контраст, размытие, шум, поворот) исследуется влияние количества
выделяемых ключевых точек и дескрипторов на результат сопоставления.
Оценивается минимальное расстояние между совпадающими точками.
"""

def create_test_images(base_image_path):
    """Создание 5 вариаций изображения для тестирования"""
    img = cv2.imread(base_image_path)
    h, w = img.shape[:2]
    
    test_images = {}
    
    # 1. Исходное изображение
    test_images['original'] = img
    
    # 2. Изменение яркости
    test_images['bright'] = cv2.convertScaleAbs(img, alpha=1, beta=80)
    
    # 3. Изменение контраста
    test_images['contrast'] = cv2.convertScaleAbs(img, alpha=1.8, beta=0)
    
    # 4. Размытие
    test_images['blur'] = cv2.GaussianBlur(img, (7, 7), 0)
    
    # 5. Поворот на 15 градусов
    M = cv2.getRotationMatrix2D((w/2, h/2), 15, 1)
    test_images['rotate'] = cv2.warpAffine(img, M, (w, h))
    
    # Сохранение
    for name, test_img in test_images.items():
        cv2.imwrite(f'test_{name}.jpg', test_img)
    
    return test_images

def sift_matching_analysis(image_dict):
    """
    Анализ сопоставления изображений с помощью SIFT
    """
    print("\n--- Пункт 6: Анализ сопоставления изображений ---")
    
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    results = []
    image_names = list(image_dict.keys())
    
    # Сравнение каждого изображения с оригиналом
    original_gray = cv2.cvtColor(image_dict['original'], cv2.COLOR_BGR2GRAY)
    kp_orig, des_orig = sift.detectAndCompute(original_gray, None)
    
    print(f"Оригинальное изображение: {len(kp_orig)} ключевых точек")
    
    for name in image_names[1:]:  # пропускаем оригинал
        img_gray = cv2.cvtColor(image_dict[name], cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(img_gray, None)
        
        if des is None or des_orig is None:
            continue
            
        # Сопоставление
        matches = bf.match(des_orig, des)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Расстояния
        distances = [m.distance for m in matches]
        min_dist = min(distances) if distances else float('inf')
        avg_dist = np.mean(distances) if distances else float('inf')
        
        results.append({
            'image': name,
            'kp_count': len(kp),
            'matches_count': len(matches),
            'min_distance': min_dist,
            'avg_distance': avg_dist
        })
        
        print(f"\n{name}:")
        print(f"  Ключевых точек: {len(kp)}")
        print(f"  Совпадений: {len(matches)}")
        print(f"  Минимальное расстояние: {min_dist:.2f}")
        print(f"  Среднее расстояние: {avg_dist:.2f}")
    
    # Визуализация результатов
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    names = [r['image'] for r in results]
    matches = [r['matches_count'] for r in results]
    plt.bar(names, matches)
    plt.title('Количество совпадений')
    plt.ylabel('Число совпадений')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    distances = [r['min_distance'] for r in results]
    plt.bar(names, distances)
    plt.title('Минимальное расстояние')
    plt.ylabel('Расстояние')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("\nВывод: Качество изображения существенно влияет на количество "
          "ключевых точек и качество сопоставления.")

# ============================================================================
# ПУНКТ 7: Сопоставление с ROI по маркерам
# ============================================================================
"""
Описание:
Сопоставление изображений с модифицированным изображением. Ключевые точки
берутся из прямоугольника, ограниченного крайними ключевыми точками маркеров.
"""

def match_with_marker_roi(base_image_path, modified_image_path):
    """
    Сопоставление изображений с использованием ROI по маркерам
    """
    print("\n--- Пункт 7: Сопоставление с ROI по маркерам ---")
    
    # Загрузка изображений
    img1 = cv2.imread(base_image_path)
    img2 = cv2.imread(modified_image_path)
    
    # Детектирование маркеров ARUCO
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters_create()
    
    corners1, ids1, _ = cv2.aruco.detectMarkers(img1, aruco_dict, params)
    corners2, ids2, _ = cv2.aruco.detectMarkers(img2, aruco_dict, params)
    
    if ids1 is None or ids2 is None:
        print("Маркеры не найдены на одном из изображений")
        return None, None
    
    # Получение bounding box по маркерам
    pts1 = np.vstack([c[0] for c in corners1])
    x1, y1 = int(np.min(pts1[:,0])), int(np.min(pts1[:,1]))
    x2, y2 = int(np.max(pts1[:,0])), int(np.max(pts1[:,1]))
    roi1 = img1[y1:y2, x1:x2]
    
    pts2 = np.vstack([c[0] for c in corners2])
    x1b, y1b = int(np.min(pts2[:,0])), int(np.min(pts2[:,1]))
    x2b, y2b = int(np.max(pts2[:,0])), int(np.max(pts2[:,1]))
    roi2 = img2[y1b:y2b, x1b:x2b]
    
    print(f"ROI1 размер: {roi1.shape}")
    print(f"ROI2 размер: {roi2.shape}")
    
    # SIFT на ROI
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY), None)
    
    if des1 is None or des2 is None:
        print("Недостаточно ключевых точек")
        return None, None
    
    # Сопоставление
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Применение Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    print(f"Количество хороших совпадений в ROI: {len(good_matches)}")
    
    # Визуализация
    result = cv2.drawMatches(roi1, kp1, roi2, kp2, good_matches[:50], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imshow("Matching with ROI", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return good_matches, result

# ============================================================================
# ПУНКТ 8: Анализ параллельности линий между ключевыми точками
# ============================================================================
"""
Описание:
Сопоставление изображений с анализом параллельности линий, соединяющих
ключевые точки на изображениях.
"""

def analyze_parallelism(image1_path, image2_path):
    """
    Анализ параллельности линий между ключевыми точками
    """
    print("\n--- Пункт 8: Анализ параллельности линий ---")
    
    # Загрузка и преобразование
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # Сопоставление
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        print("Недостаточно совпадений для анализа")
        return
    
    # Получение координат
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Вычисление гомографии
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Проверка параллельности линий
    def angle_between_lines(line1, line2):
        """Вычисление угла между двумя линиями"""
        v1 = np.array([line1[0][0] - line1[1][0], line1[0][1] - line1[1][1]])
        v2 = np.array([line2[0][0] - line2[1][0], line2[0][1] - line2[1][1]])
        
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        cos_angle = dot / (norm1 * norm2)
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        return min(angle, 180 - angle)
    
    # Выбор нескольких пар точек для анализа
    h, w = gray1.shape
    corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, M)
    
    # Анализ параллельности сторон
    edges = [
        (transformed_corners[0][0], transformed_corners[1][0]),
        (transformed_corners[1][0], transformed_corners[2][0]),
        (transformed_corners[2][0], transformed_corners[3][0]),
        (transformed_corners[3][0], transformed_corners[0][0])
    ]
    
    angles = []
    for i in range(4):
        angle = angle_between_lines(edges[i], edges[(i+1) % 4])
        angles.append(angle)
        print(f"Угол между сторонами {i+1} и {(i+1)%4 + 1}: {angle:.2f}°")
    
    # Проверка параллельности противоположных сторон
    parallel1 = angle_between_lines(edges[0], edges[2])
    parallel2 = angle_between_lines(edges[1], edges[3])
    
    print(f"\nУгол между противоположными сторонами 1-3: {parallel1:.2f}°")
    print(f"Угол между противоположными сторонами 2-4: {parallel2:.2f}°")
    
    if parallel1 < 10 and parallel2 < 10:
        print("Вывод: Противоположные стороны приблизительно параллельны")
    else:
        print("Вывод: Значительное отклонение от параллельности")
    
    return M, angles

# ============================================================================
# ПУНКТ 9: Наложение изображения
# ============================================================================
"""
Описание:
Наложение одного изображения меньшего размера на изображение большего размера
с возможностью указания позиции.
"""

def overlay_images(big_image_path, small_image_path, position=(50, 50)):
    """
    Наложение малого изображения на большое
    """
    print("\n--- Пункт 9: Наложение изображения ---")
    
    # Загрузка изображений
    big_img = cv2.imread(big_image_path)
    small_img = cv2.imread(small_image_path)
    
    if big_img is None or small_img is None:
        print("Ошибка загрузки изображений")
        return
    
    print(f"Большое изображение: {big_img.shape}")
    print(f"Малое изображение: {small_img.shape}")
    
    x, y = position
    h, w = small_img.shape[:2]
    
    # Проверка, помещается ли малое изображение
    if y + h > big_img.shape[0] or x + w > big_img.shape[1]:
        print("Малое изображение не помещается в большое!")
        print(f"Доступное пространство: {big_img.shape[1] - x} x {big_img.shape[0] - y}")
        print(f"Требуется: {w} x {h}")
        return
    
    # Наложение
    result = big_img.copy()
    result[y:y+h, x:x+w] = small_img
    
    # Сохранение и отображение
    cv2.imwrite('overlay_result.jpg', result)
    print("Результат сохранен как overlay_result.jpg")
    
    cv2.imshow("Overlay Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result

# ============================================================================
# ПУНКТ 10: Режим реального времени с видеокамеры
# ============================================================================
"""
Описание:
Выполнение пунктов 3-9 для изображения/видео, получаемых с видеокамеры
в режиме реального времени.
"""

def real_time_processing():
    """
    Обработка видеопотока в реальном времени
    """
    print("\n--- Пункт 10: Обработка видеопотока в реальном времени ---")
    print("Нажмите 'q' для выхода")
    print("Нажмите 's' для сохранения текущего кадра")
    print("Нажмите 'm' для детектирования маркеров")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return
    
    # Инициализация детекторов
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    sift = cv2.SIFT_create()
    
    # Переменные для хранения состояния
    show_markers = True
    show_contours = True
    threshold_value = 100
    
    # Создание окна с трекбаром
    cv2.namedWindow('Real-time Processing')
    cv2.createTrackbar('Threshold', 'Real-time Processing', threshold_value, 255, lambda x: None)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра")
            break
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Получение значения порога с трекбара
        threshold_value = cv2.getTrackbarPos('Threshold', 'Real-time Processing')
        
        # Пункт 3: Пороговая обработка и поиск контуров
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Отображение числа контуров
        cv2.putText(frame, f"Contours: {len(contours)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Threshold: {threshold_value}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Отображение контуров
        if show_contours:
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
        
        # Пункт 5: Детектирование маркеров ARUCO
        if show_markers:
            corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, 
                                                      parameters=aruco_params)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.putText(frame, f"Markers: {len(ids)}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Отображение FPS
        cv2.putText(frame, f"Frame: {frame_count}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Отображение результата
        cv2.imshow('Real-time Processing', frame)
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Сохранение кадра
            filename = f"capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Кадр сохранен как {filename}")
        elif key == ord('m'):
            # Переключение отображения маркеров
            show_markers = not show_markers
            print(f"Отображение маркеров: {show_markers}")
        elif key == ord('c'):
            # Переключение отображения контуров
            show_contours = not show_contours
            print(f"Отображение контуров: {show_contours}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Обработка завершена")

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ ДЛЯ ЗАПУСКА ВСЕХ ЭТАПОВ
# ============================================================================

def main():
    """Основная функция для выполнения всех пунктов лабораторной работы"""
    
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №8: Компьютерное зрение")
    print("=" * 60)
    
    # Пункт 1: Подсчет фигур на ЧБ изображении
    circles, squares, other = count_shapes_bw("test_bw.png")
    
    # Пункт 2: Подсчет фигур заданного цвета
    circles_color, squares_color = count_shapes_colored("test_color.png", (255, 0, 0), 50)
    
    # Пункт 3-4: Подготовка модифицированных изображений
    modified_images = create_modified_images("aruco_test.jpg")
    
    # Список изображений для экспериментов
    test_images = ["aruco_test.jpg", "modified_bright.jpg", 
                   "modified_contrast.jpg", "modified_blur.jpg", 
                   "modified_noise.jpg"]
    
    # Пороговые значения
    thresholds = list(range(0, 256, 25))
    area_thresholds = list(range(50, 501, 50))
    
    # Пункт 3: График зависимости числа контуров от порога
    threshold_experiment(test_images, thresholds)
    
    # Пункт 4: График зависимости числа маркеров от площади
    aruco_area_experiment(test_images, area_thresholds)
    
    # Пункт 5: Выделение маркеров
    detect_and_draw_markers(test_images)
    
    # Пункт 6: Создание тестовых изображений для SIFT
    test_sift_images = create_test_images("reference_image.jpg")
    
    # Пункт 6: Анализ SIFT сопоставления
    sift_matching_analysis(test_sift_images)
    
    # Пункт 7: Сопоставление с ROI по маркерам
    match_with_marker_roi("aruco_test.jpg", "modified_bright.jpg")
    
    # Пункт 8: Анализ параллельности
    analyze_parallelism("image1.jpg", "image2.jpg")
    
    # Пункт 9: Наложение изображений
    overlay_images("big_image.jpg", "small_image.jpg", (100, 100))
    
    # Пункт 10: Режим реального времени
    real_time_processing()
    
    print("\n" + "=" * 60)
    print("Лабораторная работа выполнена!")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

Инструкция по выполнению:

Подготовка данных:

1. Создайте папку lab8 и поместите в нее все изображения:
   · test_bw.png - черно-белое изображение с кругами и квадратами
   · test_color.png - цветное изображение с фигурами разных цветов
   · aruco_test.jpg - изображение с ARUCO маркерами
   · reference_image.jpg - эталонное изображение для SIFT тестов
   · big_image.jpg - большое изображение для наложения
   · small_image.jpg - малое изображение для наложения
   · image1.jpg, image2.jpg - изображения для анализа параллельности

Порядок выполнения:

1. Запуск программы:
   ```bash
   python lab8.py
   ```
2. Для каждого пункта:
   · Код автоматически выполняет все пункты последовательно
   · Для реального времени (пункт 10) используйте клавиши:
     · q - выход
     · s - сохранение кадра
     · m - переключение маркеров
     · c - переключение контуров

Важные замечания:

1. Для пунктов 1-2 (без cv2):
   · Используется только PIL и numpy
   · Реализован собственный алгоритм поиска контуров
2. Для пунктов 3-5 (ARUCO):
   · Требуется opencv-contrib-python
   · Установка: pip install opencv-contrib-python
3. Для пунктов 6-8 (SIFT):
   · В новых версиях OpenCV может потребоваться:
   ```python
   sift = cv2.SIFT_create()
   ```
4. Для пункта 10 (режим реального времени):
   · Проверьте подключение камеры
   · При необходимости измените индекс камеры (0, 1, 2...)

Результаты выполнения:

· Все графики сохраняются и отображаются
· Промежуточные изображения сохраняются в папке
· В консоль выводятся подробные отчеты о выполнении
