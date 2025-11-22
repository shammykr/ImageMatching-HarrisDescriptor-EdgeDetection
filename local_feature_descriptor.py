import numpy as np
import math
from PIL import Image
def compute_local_histograms(Ix, Iy, corners):
    """
    Compute normalized gradient-direction histograms at each corner pixel.
    Args:
        Ix, Iy : gradient images (2D numpy arrays)
        corners : list of (i, j) corner coordinates

    Returns:
        A list of tuples: ((i, j), hn), where hn is the normalized histogram (8 bins)
    """
    magnitude = np.sqrt(Ix**2 + Iy**2)
    angle = (np.degrees(np.arctan2(Iy, Ix)) + 360) % 360   # range [0, 360)
    H, W = Ix.shape

    descriptors = []

    for (i, j) in corners:
        # Skip corners too close to border
        if i < 4 or j < 4 or i >= H - 4 or j >= W - 4:
            continue

        # 9x9 neighborhood
        patch_mag = magnitude[i - 4:i + 5, j - 4:j + 5]
        patch_ang = angle[i - 4:i + 5, j - 4:j + 5]

        # Step 1: compute histogram h(i) for i = 0..7 (bins of 45 degrees)
        h = np.zeros(8)
        for x in range(9):
            for y in range(9):
                mag = patch_mag[x, y]
                ang = patch_ang[x, y]
                bin_idx = int(round(ang / 45.0)) % 8
                h[bin_idx] += mag

        # Step 2: find max bin index
        m = np.argmax(h)

        # Step 3: rotate histogram so that h[m] -> hn[4] (180°)
        hn = np.zeros(8)
        for k in range(8):
            hn[(4 + k) % 8] = h[(m + k) % 8]

        # Step 4: normalize
        hn /= (np.sum(hn) + 1e-6)

        descriptors.append(((i, j), hn))

    print(f"Computed {len(descriptors)} histograms for {len(corners)} corners.")
    return descriptors

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

def read_image_gray(filename, N):
    img = Image.open(filename).convert("L")
    img = img.resize((N, N))
    arr = np.array(img, dtype=float)
    print(f"Loaded image '{filename}' resized to {arr.shape}")
    return arr

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
# ---------------- Example Usage ---------------- #
if __name__ == "__main__":
    # Example: Replace with your real gradient images and detected corners
    H, W = 300, 300
    Ix = np.random.randn(H, W)
    Iy = np.random.randn(H, W)
    M_window = 11
    sigma_window = 5.5
    N = 300
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

    Ap = smooth_separable_rows_then_cols(A, M_window, sigma_window)
    Bp = smooth_separable_rows_then_cols(B, M_window, sigma_window)
    Cp = smooth_separable_rows_then_cols(C, M_window, sigma_window)
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

    R = harris_response(Ap, Bp, Cp)
    corners_mask = non_max_suppression(R, 1e-2)
    corners_list = []
    N = 300
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
    # Example: Pretend these were detected by a Harris corner detector
    corners = [(np.random.randint(5, H-5), np.random.randint(5, W-5)) for _ in range(100)]

    descriptors = compute_local_histograms(Ix, Iy, corners_list)

    # Print first 5 results as sample
    print("\nSample histograms:")
    for idx, ((i, j), hn) in enumerate(descriptors[:5]):
        print(f"pixel at (i,j)=({i},{j}) has histogram {', '.join(f'{v:.3f}' for v in hn)}")
