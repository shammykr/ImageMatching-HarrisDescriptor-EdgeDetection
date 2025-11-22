"""
Manual Gaussian Smoothing and Edge Detection (No built-in filtering)
Author: [Your Name]
Date: [Date]
Description:
    This program smooths an image using a separable 1D Gaussian filter
    (first rows, then columns), then computes gradient magnitude and
    thresholds it to mark edge pixels.
"""

from PIL import Image
import numpy as np
import math


# ------------------ Read Grayscale Image ------------------
def read_image(filename, N):
    img = Image.open(filename).convert("L")
    img = img.resize((N, N))
    f = np.array(img, dtype=float)
    print(f"Loaded image {filename} with shape {f.shape}")
    return f


# ------------------ 1D Gaussian Filter ------------------
def gaussian_1d_filter(M, sigma):
    g = np.zeros(M, dtype=float)
    center = (M - 1) // 2
    s = 0.0
    for k in range(M):
        g[k] = math.exp(-((k - center) ** 2) / (2 * sigma * sigma))
        s += g[k]
    g /= s  # normalize so sum(g) = 1
    print(f"Generated 1D Gaussian filter (σ={sigma}) -> sum={np.sum(g):.4f}")
    return g


# ------------------ Row-wise Smoothing ------------------
def smooth_rows(f, g, N, M):
    h1 = np.copy(f)
    offset = (M - 1) // 2

    for i in range(N):
        for j in range(offset, N - offset):
            s = 0.0
            for k in range(M):
                s += g[k] * f[i, j - (k - offset)]
            h1[i, j] = s
    return h1


# ------------------ Column-wise Smoothing ------------------
def smooth_cols(h1, g, N, M):
    h2 = np.copy(h1)
    offset = (M - 1) // 2

    for j in range(N):
        for i in range(offset, N - offset):
            s = 0.0
            for k in range(M):
                s += g[k] * h1[i - (k - offset), j]
            h2[i, j] = s
    return h2


# ------------------ Gradient Magnitude ------------------
def compute_gradient(h, N):
    hx = np.zeros((N, N), dtype=float)
    hy = np.zeros((N, N), dtype=float)
    grad = np.zeros((N, N), dtype=float)

    for i in range(N - 1):
        for j in range(N - 1):
            hx[i, j] = h[i + 1, j] - h[i, j]
            hy[i, j] = h[i, j + 1] - h[i, j]
            grad[i, j] = math.sqrt(hx[i, j] ** 2 + hy[i, j] ** 2)
    return grad


# ------------------ Thresholding ------------------
def threshold_edges(grad, N, thresh):
    edge = np.zeros((N, N), dtype=float)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            if grad[i, j] > thresh:
                edge[i, j] = 255
            else:
                edge[i, j] = 0
    return edge


# ------------------ Save Image ------------------
def save_image(arr, filename):
    arr = np.clip(arr, 0, 255)
    img_out = Image.fromarray(arr.astype(np.uint8))
    img_out.save(filename)
    print(f"Saved {filename}")


# ------------------ Main Program ------------------
def main():
    print("===== Gaussian Smoothing + Edge Detection =====")
    filename = "pic1grey300.jpg"
    N = int(input("Enter image size N (e.g. 300): ").strip())
    M = 9  # fixed filter size as per problem

    f = read_image(filename, N)

    # Try multiple sigma and threshold values
    sigmas = [1.0, 2.0, 3.0]
    thresholds = [20, 30, 40, 50]

    for sigma in sigmas:
        g = gaussian_1d_filter(M, sigma)
        print(f"\nApplying Gaussian smoothing with σ={sigma}...")
        h1 = smooth_rows(f, g, N, M)
        h2 = smooth_cols(h1, g, N, M)

        grad = compute_gradient(h2, N)

        for thresh in thresholds:
            print(f"Computing edges for σ={sigma}, threshold={thresh}...")
            edge = threshold_edges(grad, N, thresh)
            out_name = f"edges_sigma{sigma}_th{thresh}.png"
            save_image(edge, out_name)

    print("\nProcessing complete. Check output images for edges.")


if __name__ == "__main__":
    main()
