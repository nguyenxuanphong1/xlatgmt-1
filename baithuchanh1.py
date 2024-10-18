import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm hiển thị ảnh
def show_image(title, image):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Đọc ảnh đầu vào (ảnh xám để dễ thao tác)
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Tạo ảnh âm tính
negative_image = 255 - image

# 2. Tăng độ tương phản
alpha = 1.5  # Hệ số độ tương phản
beta = 0  # Độ sáng thêm vào
contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# 3. Biến đổi log
log_image = np.uint8(255 * (np.log1p(image) / np.log1p(255)))

# 4. Cân bằng histogram
hist_eq_image = cv2.equalizeHist(image)

# Hiển thị kết quả
show_image("Original Image", image)
show_image("Negative Image", negative_image)
show_image("Contrast Image", contrast_image)
show_image("Log Transformed Image", log_image)
show_image("Histogram Equalized Image", hist_eq_image)

# Lưu kết quả
cv2.imwrite('negative_image.jpg', negative_image)
cv2.imwrite('contrast_image.jpg', contrast_image)
cv2.imwrite('log_image.jpg', log_image)
cv2.imwrite('hist_eq_image.jpg', hist_eq_image)
