import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def smothing(f, sigma,M):
    g_x = np.zeros(M,dtype=np.float32)
    g_y = np.zeros(M,dtype=np.float32)
    center_x = M//2
    center_y = M//2

    total=0
    for i in range(M):
        g_x[i]=np.exp(-1*((i-center_x)*(i-center_x))/(2*sigma*sigma))
        total+=g_x[i]
    # Normalize g
    g_x = g_x/total

    total=0
    for i in range(M):
        g_y[i]=np.exp(-1*((i-center_y)*(i-center_y))/(2*sigma*sigma))
        total+=g_y[i]
    # Normalize g
    g_y = g_y/total

    f = np.array(f)
    N = f.shape[0]

    # Prepare output
    h_x = np.zeros([N,N], dtype=np.float64)
    for i in range((M-1)//2, N-(M-1)//2):
        for j in range(0, N):
            acc = 0.0
            for k in range(M):   
                ii = i - (k - (M-1)//2)
                acc += g_x[k] * f[ii, j]
            h_x[i,j] = acc
    h = np.zeros([N,N], dtype=np.float64)
    for i in range(0, N):
        for j in range((M-1)//2, N-(M-1)//2):
            acc = 0.0
            for k in range(M):   
                jj = j - (k - (M-1)//2)
                acc += g_y[k] * h_x[i, jj]
            h[i,j] = acc

    return h




def corner_detection(image_path,theshold=0.003):

    f = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    N=len(f)    
    h = smothing(f, sigma=2.0,M=9)
    h_disp = np.clip(np.round(h), 0, 255).astype(np.uint8)


    I_x = np.zeros([N,N], dtype=np.float64)
    I_y = np.zeros([N,N], dtype=np.float64)
    edges = []
    for i in range(0,N):
        for j in range(1,N-1):
            I_x[i,j] = abs(h[i,j+1] - h[i,j-1])
            I_x[i,j] /=10.0
    for i in range(1,N-1):
        for j in range(0,N):
            I_y[i,j] = abs(h[i+1,j] - h[i-1,j])
            I_y[i,j] /=10.0

    
    A = I_x**2
    B = I_y**2
    C = I_x * I_y

    A_smooth = smothing(A, sigma=5.5,M=11)
    B_smooth = smothing(B, sigma=5.5,M=11)
    C_smooth = smothing(C, sigma=5.5,M=11)

    R = A_smooth * B_smooth - C_smooth**2 - 0.04 * (A_smooth + B_smooth)**2

    print("R min:", np.min(R))
    print("R max:", np.max(R))
    print("R mean:", np.mean(R))

    # Choose threshold relative to max
    threshold = theshold * np.max(R)

    R_display = np.copy(f)
    corners = np.argwhere(R > threshold)

    true_corners = []

    for corner in corners:
        i, j = corner
        if(i==0 or j==0 or i==N-1 or j==N-1):
            continue
        if R[i,j] == np.max(R[i-1:i+2, j-1:j+2]):
            true_corners.append((i,j))

    # Display corners in red on original image also show the x and y components
    f_color = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
    for i,j in true_corners:
        cv2.circle(f_color, (j,i), 3, (255,0,0), 1)  # Red circle with radius 3 
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title('Original Image')
    plt.axis('off')
    plt.imshow(f, cmap='gray')
    plt.subplot(1,2,2)
    plt.title('Corners Detected')
    f_color_rgb = cv2.cvtColor(f_color, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(f_color_rgb)
    plt.show()

    #Creating histograms for each corner
    Histograms=np.zeros((len(true_corners),8))
    index =0
    for i,j in (true_corners):
        for dx in range(-1,2):
            for dy in range(-1,2):
                i_y= I_y[i+dx, j+dy]
                i_x = I_x[i+dx, j+dy]
                angle = (np.degrees(np.arctan2(i_y, i_x)) + 360.0) % 360.0
                i = int(angle // 45)
                Histograms[index,i] += 1
        index += 1

    # Rotation-normalize the histograms
    for idx in range(Histograms.shape[0]):
        row = Histograms[idx].copy()
        if row.sum() == 0:
            continue
        max_index = int(np.argmax(row))
        shift = 4 - max_index
        Histograms[idx] = np.roll(row, shift)

    return np.array(Histograms)

Histograms1 = corner_detection('pic1grey300.jpg',theshold=0.002)
Histograms2 =corner_detection('pic2grey300.jpg',theshold=0.001)

# Manually selecting the points . These are the pairs of corresponding pairs
# Pair 1: (36, 206) ↔ (80, 195)
# Pair 2: (232, 25) ↔ (251, 71)
# Pair 3: (207, 174) ↔ (215, 186)
# Pair 4: (166, 218) ↔ (180, 219)
# Pair 5: (165, 269) ↔ (177, 258)
# Pair 6: (216, 180) ↔ (224, 191)
# Pair 7: (226, 55) ↔ (240, 95)
# Pair 8: (97, 84) ↔ (138, 108)
# Pair 9: (96, 105) ↔ (135, 121)
# Pair 10: (42, 96) ↔ (96, 112)
# Pair 11: (242, 14) ↔ (258, 61)
# Pair 12: (102, 248) ↔ (127, 236)
# Pair 13: (237, 235) ↔ (235, 237)
# Pair 14: (22, 222) ↔ (66, 210)
# Pair 15: (62, 235) ↔ (96, 221)

# [x',y'] = A * [x,y] + t

pts_dst = np.array([[80, 195],
                    [251, 71],
                    [215, 186],
                    [180, 219],
                    [177, 258],
                    [224, 191],
                    [240, 95],
                    [138, 108],
                    [135, 121],
                    [96, 112],
                    [258, 61],
                    [127, 236],
                    [235, 237],
                    [66, 210],
                    [96, 221]], dtype=np.float64)

A = np.array([36, 206, 1, 0, 0, 0,
              0, 0, 0, 36, 206, 1,
              232, 25, 1, 0, 0, 0,
              0, 0, 0, 232, 25, 1,
              207, 174, 1, 0, 0, 0,
              0, 0, 0, 207, 174, 1,
              166, 218, 1, 0, 0, 0,
              0, 0, 0, 166, 218, 1,
              165, 269, 1, 0, 0, 0,
              0, 0, 0, 165, 269, 1,
              216, 180, 1, 0, 0, 0,
              0, 0, 0, 216, 180, 1,
              226, 55, 1, 0, 0, 0,
              0, 0, 0, 226, 55, 1,
              97, 84, 1, 0, 0, 0,
              0, 0, 0, 97, 84, 1,
              96, 105, 1, 0, 0, 0,
              0, 0, 0, 96, 105, 1,
              42, 96, 1, 0, 0, 0,
              0, 0, 0, 42, 96, 1,
              242, 14, 1, 0, 0, 0,
              0, 0, 0, 242, 14, 1,
              102, 248, 1, 0, 0, 0,
              0, 0, 0, 102, 248, 1,
              237, 235, 1, 0, 0, 0,
              0, 0, 0, 237, 235, 1,
              22, 222, 1, 0, 0, 0,
              0, 0, 0, 22, 222, 1,
              62, 235, 1, 0, 0, 0,
              0, 0, 0, 62, 235, 1], dtype=np.float64).reshape(-1,6)
b= pts_dst.flatten()


#inverse method

x_vec = np.linalg.inv(A.T @ A) @ (A.T @ b)
A_matrix = x_vec[:4].reshape(2,2)
t_vector = x_vec[4:].reshape(2,1)
print("Estimated Affine Transformation:")
print("A =\n", A_matrix)
print("t =\n", t_vector)