import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def histogram_equalization(img):
    # 將圖像數據攤平成一維陣列，然後計算其直方圖
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    
    # 累積直方圖的值以形成CDF
    cdf = hist.cumsum()

    # 忽略累積分布函數中的0值進行遮罩，以避免後續計算時的除零錯誤
    cdf_m = np.ma.masked_equal(cdf, 0)
    # 進行直方圖等化，線性擴展CDF範圍至[0,255]
    cdf_m = (cdf_m - cdf_m.min())*255 / (cdf_m.max() - cdf_m.min())
    # 用0填充遮罩部分，轉換為uint8類型
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # 映射到新的直方圖
    img_eq = cdf[img]

    return img_eq, hist

def calculate_CDF(hist):
    # 計算直方圖的累積分佈函數（CDF）
    cdf = hist.cumsum()
    # 正規化CDF
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min()) 

    return cdf

def add_salt_pepper_noise(img):

    salt_pepper_ratio = 0.02
    amount = 0.04

    noisy_img = np.copy(img)
    # 計算需要加雜訊的像素點數量
    num_salt = np.ceil(amount * img.size * salt_pepper_ratio)
    num_pepper = np.ceil(amount * img.size * (1.0 - salt_pepper_ratio))

    # 加入鹽雜訊
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 1

    # 加入胡椒雜訊
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 0

    return noisy_img

def mean_filter(img, kernel_size=3):
    # 創建一個均值核 (Mean kernel)
    mean_kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)

    # 初始化輸出圖像
    mean_filtered = np.zeros_like(img)

    # 檢查圖像是RGB還是灰階
    if len(img.shape) == 3 and img.shape[2] == 3:
        # 對於RGB圖像，單獨對每個通道應用濾波器
        for channel in range(3):
            mean_filtered[:, :, channel] = convolve2d(img[:, :, channel], mean_kernel, mode='same', boundary='fill', fillvalue=0)
    
    elif len(img.shape) == 2:
        # 對於灰階圖像，直接應用濾波器
        mean_filtered = convolve2d(img, mean_kernel, mode='same', boundary='fill', fillvalue=0)

    else:
        # 如果圖像既不是RGB也不是灰階，引發錯誤
        raise ValueError("Input image must be an RGB image or a grayscale image.")

    return mean_filtered

def median_filter(img, kernel_size=3):
    k = kernel_size // 2
    median_filtered = np.zeros_like(img)
    
    # 檢查圖像是RGB還是灰階
    if len(img.shape) == 3 and img.shape[2] == 3:
        # 處理RGB圖像
        for channel in range(3):
            for i in range(k, img.shape[0] - k):
                for j in range(k, img.shape[1] - k):
                    k_region = img[i - k:i + k + 1, j - k:j + k + 1, channel]
                    median_filtered[i, j, channel] = np.median(k_region)

    elif len(img.shape) == 2:
        # 處理灰階圖像
        for i in range(k, img.shape[0] - k):
            for j in range(k, img.shape[1] - k):
                k_region = img[i - k:i + k + 1, j - k:j + k + 1]
                median_filtered[i, j] = np.median(k_region)
    else:
        raise ValueError("Input image must be an RGB image or a grayscale image.")
        
    return median_filtered

def gaussian_filter(img, kernel_size=3, sigma=1.4):
    k = kernel_size // 2

    kernel_range = range(-int((kernel_size - 1) / 2), int((kernel_size + 1) / 2))
    gaussian_kernel_2d = np.array([[1 / (2 * np.pi * sigma**2) * 
                                    np.exp(- (i**2 + j**2) / (2 * sigma**2)) 
                                    for i in kernel_range] for j in kernel_range])
    gaussian_kernel_2d /= np.sum(gaussian_kernel_2d)

    gaussian_filtered = np.zeros_like(img)
    
    # 處理灰階圖像
    for i in range(k, img.shape[0] - k):
        for j in range(k, img.shape[1] - k):
            k_region = img[i - k:i + k + 1, j - k:j + k + 1]
            gaussian_filtered[i, j] = np.sum(k_region * gaussian_kernel_2d)
    
    return gaussian_filtered

def canny_edge_detection(img):
    length, width, _ = img.shape
    
    img = 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]
    
    img = gaussian_filter(img, kernel_size=3, sigma=1.4)

    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Gx = np.zeros_like(img, dtype=np.float32)
    Gy = np.zeros_like(img, dtype=np.float32)

    for i in range(1, length - 1):
        for j in range(1, width - 1):
            k_region = img[i - 1:i + 2, j - 1:j + 2]
            Gx[i, j] = np.sum(k_region * Kx)
            Gy[i, j] = np.sum(k_region * Ky)

    G_magnitude = np.sqrt((Gx ** 2)+(Gy ** 2))
    G_angle = np.arctan2(Gy, Gx) * (180 / np.pi)  # Convert angle to degrees

    # Adjust angle values to be between 0 and 180 degrees
    G_angle = np.mod(G_angle + 180, 180)

    M, N = G_magnitude.shape
    Z = np.zeros((M, N), dtype=np.float32)
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            # Determine the neighbouring pixels to inspect based on gradient direction
            if (0 <= G_angle[i,j] < 22.5) or (157.5 <= G_angle[i,j] <= 180):
                q = G_magnitude[i, j+1]
                r = G_magnitude[i, j-1]
            elif (22.5 <= G_angle[i,j] < 67.5):
                q = G_magnitude[i+1, j-1]
                r = G_magnitude[i-1, j+1]
            elif (67.5 <= G_angle[i,j] < 112.5):
                q = G_magnitude[i+1, j]
                r = G_magnitude[i-1, j]
            elif (112.5 <= G_angle[i,j] < 157.5):
                q = G_magnitude[i-1, j-1]
                r = G_magnitude[i+1, j+1]

                # Suppress non-maximum pixels
            if (G_magnitude[i,j] >= q) and (G_magnitude[i,j] >= r):
                Z[i,j] = G_magnitude[i,j]
            else:
                Z[i,j] = 0


    high_threshold_ratio = 0.2
    low_threshold_ratio = high_threshold_ratio * 0.5
 
    weak_val = 25
    strong_val = 255

    high_threshold = Z.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio
    
    M, N = Z.shape
    res = np.zeros((M,N), dtype=np.uint8)
    
    # Identify strong, weak, and non-relevant pixels
    strong_i, strong_j = np.where(Z >= high_threshold)
    weak_i, weak_j = np.where((Z <= high_threshold) & (Z >= low_threshold))
    
    # Set the identified pixels in the result image
    res[strong_i, strong_j] = strong_val
    res[weak_i, weak_j] = weak_val
    
    # Edge Tracking by Hysteresis
    # Pixels with weak value that are connected to strong value pixels will be considered as strong
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (res[i, j] == weak_val):
                if ((res[i+1, j-1] == strong_val) or (res[i+1, j] == strong_val) or
                    (res[i+1, j+1] == strong_val) or (res[i, j-1] == strong_val) or
                    (res[i, j+1] == strong_val) or (res[i-1, j-1] == strong_val) or
                    (res[i-1, j] == strong_val) or (res[i-1, j+1] == strong_val)):
                    res[i, j] = strong_val
    
    return res

def Histogram_Equalization():
    img = cv2.imread('road.jpg', cv2.IMREAD_GRAYSCALE)

    #########################手刻法進行直方圖等化#########################

    # 使用手刻法進行直方圖等化
    img_eq, hist_before = histogram_equalization(img)

    # 計算等化後的直方圖
    hist_after, bins = np.histogram(img_eq.flatten(), 256, [0, 256])

    #####################################################################

    '''
    ###########################OpenCV function###########################

    # 計算原始圖片的直方圖
    hist_before = cv2.calcHist([img], [0], None, [256], [0, 256])

    # 進行直方圖等化
    img_eq = cv2.equalizeHist(img)
    # 用CLAHE均衡
    clahe = cv2.createCLAHE()
    #img_eq = clahe.apply(img)

    # 計算等化後的直方圖
    hist_after = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

    #####################################################################
    '''

    cdf_before = calculate_CDF(hist_before)
    cdf_after = calculate_CDF(hist_after)

    # 使用matplotlib顯示圖片與直方圖
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    plt.subplot(2, 3, 2)
    plt.bar(range(256), hist_before.flatten(), width=1)
    plt.title("Histogram of Original Image")

    plt.subplot(2, 3, 3)
    plt.plot(cdf_before, color='r')
    plt.title("CDF of Original Image")

    plt.subplot(2, 3, 4)
    plt.imshow(img_eq, cmap='gray')
    plt.title("Equalized Image")

    plt.subplot(2, 3, 5)
    plt.bar(range(256), hist_after.flatten(), width=1)
    plt.title("Histogram of Equalized Image")

    plt.subplot(2, 3, 6)
    plt.plot(cdf_after, color='r')
    plt.title("CDF of Equalized Image")

    plt.tight_layout()
    plt.show()

def salt_pepper_img():
    img = cv2.imread("Patrick.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 加入胡椒鹽雜訊
    noisy_img = add_salt_pepper_noise(img)

    # 顯示原圖與加入雜訊後的圖片
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(noisy_img)
    plt.title('Image with Salt and Pepper Noise')
    plt.axis('off')

    plt.show()

def filter_img():
    # 實現使用OpenCV函數的Mean filter和Median filter

    img = cv2.imread("Patrick.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 加入胡椒鹽雜訊
    noisy_img = add_salt_pepper_noise(img)
    
    '''
    ###########################OpenCV function###########################

    # Mean filter - 平均濾波器，使用cv2.blur()函數
    mean_filtered = cv2.blur(noisy_img, (3,3))

    # Median filter - 中值濾波器，使用cv2.medianBlur()函數
    median_filtered = cv2.medianBlur(noisy_img, 3)

    #####################################################################
    '''
    
    ############################手刻法進行濾波############################

    mean_filtered = mean_filter(img, kernel_size = 5)

    median_filtered = median_filter(img, kernel_size = 5)

    #####################################################################

    # 展示原圖以及經由filter得圖
    plt.figure(figsize=(15, 5))

    # 原圖
    plt.subplot(1, 3, 1)
    plt.imshow(noisy_img)
    plt.title('Original Image')
    plt.axis('off')

    # Mean filter 的圖
    plt.subplot(1, 3, 2)
    plt.imshow(mean_filtered)
    plt.title('Mean Filtered Image')
    plt.axis('off')

    # Median filter 的圖
    plt.subplot(1, 3, 3)
    plt.imshow(median_filtered)
    plt.title('Median Filtered Image')
    plt.axis('off')

    plt.show()

def canny_img():
    img = cv2.imread("road.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    canny_img = canny_edge_detection(img)
    

    # 顯示原圖與加入雜訊後的圖片
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(canny_img, cmap='gray')
    plt.title('Image with Canny Edge Detection')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    #調用要用的功能
    #Histogram_Equalization()
    #salt_pepper_img()
    #filter_img()
    canny_img()