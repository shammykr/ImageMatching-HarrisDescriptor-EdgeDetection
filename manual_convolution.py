"""
Manual 2D Convolution (No built-in filtering)
Author: [Your Name]
Date: [Date]
Description:
    This program performs 2D convolution on an NxN grayscale image using an MxM filter.
    All computations are done using explicit pixel intensity values (no built-in functions).
    The filter can be:
        (1) Read from a file, or
        (2) A Gaussian filter with sigma = M/4.0
    The program outputs the convolved image and saves it as 'output.png'.
"""

from PIL import Image
import numpy as np
import math


# ------------------ Image Reading ------------------
def read_image(filename, N):
    """Read grayscale image and resize to NxN."""
    img = Image.open(filename).convert("L")
    img = img.resize((N, N))
    f = np.array(img, dtype=float)
    return f


# ------------------ Gaussian Filter ------------------
def generate_gaussian_filter(M):
    """Generate an MxM Gaussian filter with sigma = M/4.0 and normalize it."""
    sigma = M / 4.0
    g = [[0 for _ in range(M)] for _ in range(M)]
    c = (M - 1) // 2
    s = 0.0

    for i in range(M):
        for j in range(M):
            g[i][j] = math.exp(-((i - c)**2 + (j - c)**2) / (2 * sigma**2))
            s += g[i][j]

    # Normalize so sum(g) = 1
    for i in range(M):
        for j in range(M):
            g[i][j] /= s

    return g


# ------------------ Read Filter from File ------------------
def read_filter_from_file(filename, M):
    """Read an MxM filter from a text file."""
    g = []
    with open(filename, 'r') as f:
        for _ in range(M):
            row = [float(x) for x in f.readline().split()]
            g.append(row)
    return g


# ------------------ Manual Convolution ------------------
def convolve(f, g, N, M):
    """Perform manual 2D convolution using explicit pixel operations."""
    h = [[0 for _ in range(N)] for _ in range(N)]
    offset = (M - 1) // 2

    for i in range(offset, N - offset):
        for j in range(offset, N - offset):
            s = 0.0
            for k in range(M):
                for l in range(M):
                    # Compute weighted sum from neighborhood
                    s += g[k][l] * f[i - (k - offset)][j - (l - offset)]
            h[i][j] = s

    return np.array(h)


# ------------------ Save Output Image ------------------
def save_output_image(h, filename):
    """Clip, convert to uint8, and save/display output image."""
    h = np.clip(h, 0, 255)
    img_out = Image.fromarray(h.astype(np.uint8))
    img_out.save(filename)
    img_out.show()


# ------------------ Main ------------------
def main():
    print("===== Manual 2D Convolution =====")
    filename = "pic1grey300.jpg"
    N = int(input("Enter image size N (e.g. 300): ").strip())
    M = int(input("Enter filter size M (odd number, e.g. 9): ").strip())

    if M % 2 == 0:
        print("Error: M must be an odd number.")
        return

    print("Choose filter option:")
    print("1 - Read filter from file")
    print("2 - Use Gaussian filter (σ = M/4.0)")
    option = int(input("Enter option (1 or 2): ").strip())

    # Read image
    f = read_image(filename, N)

    # Create or load filter
    if option == 1:
        filter_file = input("Enter filter filename: ").strip()
        g = read_filter_from_file(filter_file, M)
    elif option == 2:
        g = generate_gaussian_filter(M)
    else:
        print("Invalid option.")
        return

    print("Performing convolution... please wait.")
    h = convolve(f, g, N, M)

    # Border pixels set to zero
    offset = (M - 1) // 2
    h[:offset, :] = 0
    h[-offset:, :] = 0
    h[:, :offset] = 0
    h[:, -offset:] = 0

    # Save and display result
    if option == 1:
        save_output_image(h, "manual_filter_output.png")
    else:
        save_output_image(h, "output.png")

    print("Convolution complete. Output saved as 'output.png'.")


if __name__ == "__main__":
    main()
