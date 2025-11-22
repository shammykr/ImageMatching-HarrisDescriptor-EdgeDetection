"""
Harris corner detector + local 9x9 histogram descriptor
Author: [Your Name]
Date: [Date]

Usage:
    python harris_descriptor.py

This script:
 - Reads a grayscale NxN image (user supplies filename and N)
 - Smooths image with 9x9 separable Gaussian (sigma=2.0)
 - Computes gradients Ix, Iy (and divides by 10)
 - Forms A=Ix^2, B=Iy^2, C=Ix*Iy and smooths them with 11x11 separable Gaussian (sigma=5.5)
 - Computes Harris response R = det(M) - 0.04 * trace(M)^2
 - Allows auto threshold (half max R) or manual threshold
 - Performs non-max suppression and marks corner points
 - Superimposes corners (3x3 region set to 255) on the original image and saves it
 - For each corner, computes 9x9 local gradient-direction histogram descriptor:
     - directions quantized to 8 bins (0,45,...,315)
     - histogram entries incremented by gradient magnitude
     - rotate histogram so its max ends up at index 4 (180 deg)
     - normalize histogram to sum 1.0
 - Prints corner coordinates and their descriptor histograms
"""

from PIL import Image
import numpy as np
import math

# ----------------- Utilities -----------------
def read_image_gray(filename, N):
    img = Image.open(filename).convert("L")
    img = img.resize((N, N))
    arr = np.array(img, dtype=float)
    print(f"Loaded image '{filename}' resized to {arr.shape}")
    return arr

def save_image_from_array(arr, filename):
    arr = np.clip(arr, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(filename)
    print(f"Saved image: {filename}")

# ----------------- Gaussian 1D generator -----------------
def gaussian_1d(M, sigma):
    center = (M - 1) // 2
    g = [0.0] * M
    s = 0.0
    for k in range(M):
        val = math.exp(-((k - center) ** 2) / (2.0 * sigma * sigma))
        g[k] = val
        s += val
    # normalize
    for k in range(M):
        g[k] /= s
    # return as python list for explicit loops
    return g

# ----------------- Separable smoothing (explicit loops) -----------------
def smooth_separable_rows_then_cols(f, M, sigma):
    """Smooth f (NxN numpy array) with separable 1D Gaussian of size M and sigma.
       Keeps border pixels as original (only smooth interior pixels).
       Returns smoothed array (numpy float).
    """
    N = f.shape[0]
    g = gaussian_1d(M, sigma)
    offset = (M - 1) // 2

    # initialize h1 = f (so border pixels remain original)
    h1 = np.array(f, dtype=float)
    # row-wise
    for i in range(N):
        for j in range(offset, N - offset):
            s = 0.0
            for k in range(M):
                x = j - (k - offset)
                # x will be in range [0, N-1] because j in [offset, N-offset-1]
                s += g[k] * f[i, x]
            h1[i, j] = s

    # now column-wise: init h2 = h1 to preserve border pixels
    h2 = np.array(h1, dtype=float)
    for j in range(N):
        for i in range(offset, N - offset):
            s = 0.0
            for k in range(M):
                y = i - (k - offset)
                s += g[k] * h1[y, j]
            h2[i, j] = s

    return h2

# ----------------- Compute gradients (explicit) -----------------
def compute_gradients(h):
    """Compute Ix and Iy using hx(i,j)=h(i+1,j)-h(i,j) and hy(i,j)=h(i,j+1)-h(i,j)
       h is NxN numpy float array. Return Ix, Iy arrays of shape NxN,
       with borders left as 0.
    """
    N = h.shape[0]
    Ix = np.zeros((N, N), dtype=float)
    Iy = np.zeros((N, N), dtype=float)

    # compute for interior indices (i from 0..N-2, j 0..N-2)
    for i in range(0, N - 1):
        for j in range(0, N - 1):
            hx = h[i + 1, j] - h[i, j]    # derivative along i
            hy = h[i, j + 1] - h[i, j]    # derivative along j
            Ix[i, j] = hx
            Iy[i, j] = hy

    # divide by 10 as specified
    for i in range(N):
        for j in range(N):
            Ix[i, j] = Ix[i, j] / 10.0
            Iy[i, j] = Iy[i, j] / 10.0

    return Ix, Iy

# ----------------- Harris response -----------------
def harris_response(Ap, Bp, Cp):
    """Given smoothed A', B', C' arrays (numpy floats), compute R for each pixel:
       R = det(M) - 0.04 * (trace(M))^2 where M = [[A', C'], [C', B']]
    """
    N = Ap.shape[0]
    R = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            a = Ap[i, j]
            b = Bp[i, j]
            c = Cp[i, j]
            det = a * b - c * c
            trace = a + b
            R[i, j] = det - 0.04 * (trace * trace)
    return R

# ----------------- Non-maximum suppression -----------------
def non_max_suppression(R, threshold):
    """Return boolean array corners where R > threshold and R is local max among 8 neighbors.
       Border pixels will be False.
    """
    N = R.shape[0]
    corners = np.zeros((N, N), dtype=bool)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            val = R[i, j]
            if val <= threshold:
                continue
            # compare to 8 neighbors
            is_local_max = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    if R[i + di, j + dj] >= val:
                        is_local_max = False
                        break
                if not is_local_max:
                    break
            if is_local_max:
                corners[i, j] = True
    return corners

# ----------------- Superimpose corners on original image -----------------
def overlay_corners_on_image(orig, corners):
    """orig: NxN float array (0..255). corners: boolean NxN. For each True corner,
       set a 3x3 region centered at that pixel to 255 in copy of orig.
    """
    out = np.array(orig, dtype=float)
    N = out.shape[0]
    for i in range(N):
        for j in range(N):
            if corners[i, j]:
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        x = i + di
                        y = j + dj
                        if 0 <= x < N and 0 <= y < N:
                            out[x, y] = 255.0
    return out

# ----------------- Descriptor: 9x9 local rotated histogram -----------------
def compute_descriptor_at_pixel(Ix, Iy, i0, j0):
    """Compute 8-bin histogram descriptor for 9x9 patch centered at (i0,j0).
       Uses angle = atan2(Ix, Iy) in radians (per problem statement),
       angle mapped to 0..360 degrees, quantized to 8 bins of width 45 deg.
       Increment histogram bin by gradient magnitude sqrt(Ix^2 + Iy^2).
       Then rotate histogram so the max bin moves to index 4 (180 deg),
       and normalize so sum = 1.0.
       Returns hn list of 8 floats.
    """
    N = Ix.shape[0]
    half = 4  # 9x9 -> offset 4
    h = [0.0] * 8
    for di in range(-half, half + 1):
        for dj in range(-half, half + 1):
            i = i0 + di
            j = j0 + dj
            if not (0 <= i < N and 0 <= j < N):
                continue
            gx = Ix[i, j]
            gy = Iy[i, j]
            # Use atan2(Ix, Iy) per instruction (note unusual ordering)
            ang = math.atan2(gx, gy)  # returns -pi..pi
            if ang < 0:
                ang += 2.0 * math.pi
            deg = ang * 180.0 / math.pi  # 0..360
            binf = deg / 45.0
            b = int(round(binf)) % 8
            mag = math.hypot(gx, gy)
            h[b] += mag

    # find index of max
    m = 0
    vmax = h[0]
    for idx in range(1, 8):
        if h[idx] > vmax:
            vmax = h[idx]
            m = idx

    # rotate so h[m] becomes hn[4]
    hn = [0.0] * 8
    for i in range(8):
        src_idx = (m + i) % 8
        dest_idx = (4 + i) % 8
        hn[dest_idx] = h[src_idx]

    # normalize histogram to sum 1.0 if sum > 0
    s = sum(hn)
    if s > 0:
        for i in range(8):
            hn[i] /= s
    return hn

# ----------------- Main routine -----------------
def main():
    print("===== Harris Corner Detector + Descriptor =====")
    filename = "pic1grey300.jpg"
    N = int(input("Enter image size N (e.g. 300): ").strip())

    # Step 1: Smooth image with 9x9 separable Gaussian sigma=2.0
    M_smooth = 9
    sigma_smooth = 2.0
    f = read_image_gray(filename, N)
    print("Smoothing image with 9x9 separable Gaussian (sigma=2.0)...")
    h_sm = smooth_separable_rows_then_cols(f, M_smooth, sigma_smooth)

    # Step 2: Compute gradients Ix, Iy = hx, hy and divide by 10
    print("Computing gradients Ix, Iy and dividing by 10...")
    Ix, Iy = compute_gradients(h_sm)

    # Step 3: compute A,B,C
    N = f.shape[0]
    A = np.zeros((N, N), dtype=float)
    B = np.zeros((N, N), dtype=float)
    C = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            a = Ix[i, j] * Ix[i, j]
            b = Iy[i, j] * Iy[i, j]
            c = Ix[i, j] * Iy[i, j]
            A[i, j] = a
            B[i, j] = b
            C[i, j] = c

    # Step 4: Smooth A,B,C with 11x11 Gaussian sigma = 5.5
    M_window = 11
    sigma_window = 5.5
    print("Smoothing A, B, C with 11x11 separable Gaussian (sigma=5.5)...")
    Ap = smooth_separable_rows_then_cols(A, M_window, sigma_window)
    Bp = smooth_separable_rows_then_cols(B, M_window, sigma_window)
    Cp = smooth_separable_rows_then_cols(C, M_window, sigma_window)

    # Step 5 & 6: Compute R = det - 0.04 * trace^2
    print("Computing Harris response R...")
    R = harris_response(Ap, Bp, Cp)

    # Print R stats and suggest a threshold
    R_min = float(np.min(R))
    R_max = float(np.max(R))
    print(f"R min = {R_min:.6e}, R max = {R_max:.6e}")
    suggested = R_max * 0.5
    print(f"Suggested threshold (half max R) = {suggested:.6e}")

    # Ask user to accept suggested threshold or provide one
    choice = input("Use suggested threshold? (y/n) [y]: ").strip().lower()
    if choice in ("", "y", "yes"):
        thresh = suggested
    else:
        val = input("Enter threshold value for R (a floating number, e.g. 1e-4): ").strip()
        thresh = float(val)

    # Step 7 & 8: Threshold R and non-max suppression
    corners_mask = non_max_suppression(R, thresh)

    # Create overlay image: original with corners (3x3 = 255)
    overlay = overlay_corners_on_image(f, corners_mask)
    out_overlay_name = "harris_corners_overlay.png"
    save_image_from_array(overlay, out_overlay_name)

    # Print corner coordinates and compute descriptor for each
    print("\nCorner pixels and their 8-bin rotated normalized histograms:")
    corners_list = []
    for i in range(N):
        for j in range(N):
            if corners_mask[i, j]:
                # ensure we can extract 9x9 descriptor centered at (i,j)
                if i - 4 < 0 or i + 4 >= N or j - 4 < 0 or j + 4 >= N:
                    # skip corners too close to border for 9x9 window
                    print(f"corner at ({i},{j}) too close to border for 9x9 descriptor; skipping descriptor")
                    continue
                hn = compute_descriptor_at_pixel(Ix, Iy, i, j)
                # Print as requested: pixel at (i,j)=(i,j) has histogram ...
                hist_str = ", ".join(f"{val:.6f}" for val in hn)
                print(f"pixel at (i,j)=({i},{j}) has histogram {hist_str}")
                corners_list.append(((i, j), hn))

    if len(corners_list) == 0:
        print("No corners detected with the selected threshold. Try lowering the threshold.")

    print("\nDone. Outputs:")
    print(f" - Overlay image with corners: {out_overlay_name}")
    print(" - Corner coordinates and descriptors printed above.")

if __name__ == "__main__":
    main()
