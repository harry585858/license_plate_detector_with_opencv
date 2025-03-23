import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

# 상수 정의
MAX_RATIO = 5
ALLOWED_ERROR = 0.1
NUM_OF_INT = 6  # 번호판 숫자의 개수

def preprocess_image(image_path):
    """이미지를 읽고 그레이스케일 및 이진화 처리"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지 파일을 불러올 수 없습니다: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return img, binary

def find_contours(binary_img):
    """이진화된 이미지에서 컨투어 찾기"""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours(contours):
    """컨투어를 필터링하여 유효한 박스만 추출"""
    approved_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= h and w * MAX_RATIO > h and w > 10 and h > 10:
            approved_boxes.append({
                'contour': contour,
                'x': x, 'y': y,
                'w': w, 'h': h,
                'cx': x + (w / 2), 'cy': y + (h / 2)
            })
    return approved_boxes

def group_similar_boxes(approved_boxes):
    """크기와 비율이 비슷한 박스들을 그룹화"""
    grouped_boxes = []
    while approved_boxes:
        base_box = approved_boxes.pop(0)
        group = [base_box]
        remove_indices = []
        for i, other_box in enumerate(approved_boxes):
            width_diff = abs(base_box['w'] - other_box['w']) / base_box['w']
            height_diff = abs(base_box['h'] - other_box['h']) / base_box['h']
            if width_diff < ALLOWED_ERROR and height_diff < ALLOWED_ERROR:
                group.append(other_box)
                remove_indices.append(i)
        for index in sorted(remove_indices, reverse=True):
            del approved_boxes[index]
        if len(group) >= NUM_OF_INT:
            grouped_boxes.append(group)
    return grouped_boxes

def draw_boxes(img, grouped_boxes):
    """그룹화된 박스를 이미지에 그리기"""
    for group in grouped_boxes:
        for box in group:
            cv2.rectangle(img, (box['x'], box['y']), (box['x'] + box['w'], box['y'] + box['h']), (255, 0, 255), 2)
    return img

def extract_text_from_boxes(img, grouped_boxes):
    """그룹화된 박스에서 텍스트 추출"""
    results = []
    for group in grouped_boxes:
        for box in group:
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            roi = img[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi, lang='kor', config='--psm 7 --oem 0')
            results.append(text.strip())
    return results

def main():
    original_file = input('File name/파일이름 > ')
    if original_file == '':
        original_file = 'input.jpg'

    try:
        img, binary = preprocess_image(original_file)
    except FileNotFoundError as e:
        print(e)
        return

    contours = find_contours(binary)
    approved_boxes = filter_contours(contours)
    grouped_boxes = group_similar_boxes(approved_boxes)
    result_img = draw_boxes(img.copy(), grouped_boxes)

    # 결과 출력
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title('Detected Number Plate Characters')
    plt.show()

    # 텍스트 추출
    extracted_texts = extract_text_from_boxes(img, grouped_boxes)
    for text in extracted_texts:
        print(f'추출된 텍스트: {text}')

if __name__ == "__main__":
    main()