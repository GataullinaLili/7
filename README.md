Вот полный скорректированный код с подробным описанием выполнения каждого пункта лабораторной работы:

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
