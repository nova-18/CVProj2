import numpy as np
from concurrent.futures import ThreadPoolExecutor

class LucasKanadeTracker:
    def __init__(self, window_size=15, marker_points=None,
                 harris_k=0.04, harris_threshold_rel=0.01, nms_size=3):
        """
        Initialize the tracker.

        Parameters:
          window_size         : Size of the patch for the Lucas–Kanade computation.
          marker_points       : List of tuples (x, y). If provided, marker‐based
                                tracking is used. Otherwise markerless detection is performed.
          harris_k            : Harris detector free parameter.
          harris_threshold_rel: Threshold relative to the maximum Harris response.
          nms_size            : Window size for non-maximum suppression.
        """
        self.window_size = window_size
        self.marker_points = marker_points
        self.harris_k = harris_k
        self.harris_threshold_rel = harris_threshold_rel
        self.nms_size = nms_size

        # Sobel kernels for manual gradient computation.
        self.sobel_x = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]], dtype=np.float32)
        self.sobel_y = np.array([[1,  2,  1],
                                 [0,  0,  0],
                                 [-1, -2, -1]], dtype=np.float32)

    def to_gray(self, image):
        """
        Convert input image to grayscale if it is colored.
        
        Parameters:
          image: 2D or 3D numpy array.
          
        Returns:
          A 2D grayscale numpy array.
        """
        if image.ndim == 3 and image.shape[2] == 3:
            # Use standard luminosity coefficients.
            return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
        return image.astype(np.float32)

    def convolve2d(self, image, kernel):
        """
        Optimized vectorized 2D convolution using stride tricks.
        
        Parameters:
          image : 2D numpy array.
          kernel: 2D numpy array.
          
        Returns:
          convolved image.
        """
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
        pad_h, pad_w = ker_h // 2, ker_w // 2

        # Pad the image with zeros.
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        # Define the shape of the sliding windows.
        shape = (img_h, img_w, ker_h, ker_w)
        strides = (padded.strides[0], padded.strides[1],
                   padded.strides[0], padded.strides[1])
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

        # Perform element-wise multiplication and sum over the last two axes.
        conv_out = np.tensordot(windows, kernel, axes=([2, 3], [0, 1]))
        return conv_out

    def compute_gradients(self, image):
        """
        Compute image gradients using the vectorized convolve2d routine.
        
        Parameters:
          image: 2D grayscale numpy array.
        
        Returns:
          Ix, Iy: Gradients along x and y directions.
        """
        image = image.astype(np.float32)
        Ix = self.convolve2d(image, self.sobel_x)
        Iy = self.convolve2d(image, self.sobel_y)
        return Ix, Iy

    def lucas_kanade_point(self, I1, I2, point):
        """
        Compute the displacement (u, v) for a single point between frames I1 and I2.
        
        Parameters:
          I1, I2: 2D grayscale images.
          point : (x, y) coordinate in I1.
        
        Returns:
          Displacement vector (u, v).
        """
        half_w = self.window_size // 2
        x, y = int(round(point[0])), int(round(point[1]))

        # Define window boundaries safely.
        x_start = max(x - half_w, 0)
        y_start = max(y - half_w, 0)
        x_end   = min(x + half_w + 1, I1.shape[1])
        y_end   = min(y + half_w + 1, I1.shape[0])

        # Extract windows.
        patch1 = I1[y_start:y_end, x_start:x_end]
        patch2 = I2[y_start:y_end, x_start:x_end]

        # Compute gradients on patch from I1.
        Ix_patch, Iy_patch = self.compute_gradients(patch1)
        It_patch = (patch2 - patch1).astype(np.float32)

        # Vectorize gradients.
        Ix_vec = Ix_patch.flatten()
        Iy_vec = Iy_patch.flatten()
        It_vec = It_patch.flatten()

        # Formulate A and b from I_x*u + I_y*v = -I_t.
        A = np.vstack((Ix_vec, Iy_vec)).T  # shape (n,2)
        b = -It_vec.reshape(-1, 1)           # shape (n,1)

        ATA = A.T @ A
        epsilon = 1e-4
        ATA += epsilon * np.eye(2)
        nu = np.linalg.inv(ATA) @ A.T @ b
        u, v = nu.ravel()
        return (u, v)

    def harris_corner_detection(self, image):
        """
        Perform Harris corner detection with vectorized non-maximum suppression.
        
        Parameters:
          image: 2D grayscale numpy array.
        
        Returns:
          List of corner coordinates (x, y).
        """
        Ix, Iy = self.compute_gradients(image)
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # Use the optimized convolution to sum products in a window.
        kernel = np.ones((3, 3), dtype=np.float32)
        Sxx = self.convolve2d(Ixx, kernel)
        Syy = self.convolve2d(Iyy, kernel)
        Sxy = self.convolve2d(Ixy, kernel)

        # Compute the Harris response.
        detM = Sxx * Syy - Sxy * Sxy
        traceM = Sxx + Syy
        R = detM - self.harris_k * (traceM ** 2)

        # Threshold the response.
        threshold = self.harris_threshold_rel * R.max()
        R_thresh = np.where(R >= threshold, R, 0)

        # Vectorized non-maximum suppression.
        nms_half = self.nms_size // 2
        padded = np.pad(R_thresh, nms_half, mode='constant', constant_values=0)
        shape = (R_thresh.shape[0], R_thresh.shape[1], self.nms_size, self.nms_size)
        strides = (padded.strides[0], padded.strides[1],
                   padded.strides[0], padded.strides[1])
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        window_max = np.max(windows, axis=(2, 3))
        mask = (R_thresh == window_max) & (R_thresh > 0)

        ys, xs = np.nonzero(mask)
        corners = list(zip(xs, ys))
        return corners

    def track(self, I1, I2, parallel=True):
        """
        Compute optical flow from I1 to I2 at selected feature points.
        Converts colored images to grayscale if necessary.
        
        Parameters:
          I1, I2: Two consecutive images (grayscale or colored).
          parallel: If True, flow computations for individual feature points are performed in parallel.
        
        Returns:
          A list of tuples: ((x,y), (u,v))
        """
        # Convert images to grayscale if colored.
        I1 = self.to_gray(I1)
        I2 = self.to_gray(I2)
        
        if self.marker_points is not None:
            points = self.marker_points
        else:
            points = self.harris_corner_detection(I1)

        flow_vectors = []
        if parallel and len(points) > 0:
            # Use ThreadPoolExecutor to process each point in parallel.
            with ThreadPoolExecutor() as executor:
                results = executor.map(lambda pt: (pt, self.lucas_kanade_point(I1, I2, pt)), points)
                flow_vectors = list(results)
        else:
            for point in points:
                flow_vectors.append((point, self.lucas_kanade_point(I1, I2, point)))
        return flow_vectors

def KLT():
    pass

# -----------------------------
# Example usage:
# -----------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create synthetic images.
    # I1 is a 100x100 image with a white square.
    # I2 translates the square slightly.
    I1 = np.zeros((100, 100, 3), dtype=np.float32)
    I2 = np.zeros((100, 100, 3), dtype=np.float32)
    I1[30:50, 30:50] = [255, 255, 255]
    I2[32:52, 33:53] = [255, 255, 255]

    # Marker-based tracking (manual marker).
    markers = [(40, 40)]
    tracker_mb = LucasKanadeTracker(window_size=15, marker_points=markers)
    flows_mb = tracker_mb.track(I1, I2, parallel=True)
    print("Marker-based tracking:")
    for (x, y), (u, v) in flows_mb:
        print(f"Point ({x}, {y}) moved by ({u:.2f}, {v:.2f})")

    # Markerless tracking (automatic feature detection).
    tracker_ml = LucasKanadeTracker(window_size=15, marker_points=None)
    flows_ml = tracker_ml.track(I1, I2, parallel=True)
    print("\nMarkerless tracking:")
    for (x, y), (u, v) in flows_ml:
        print(f"Point ({x}, {y}) moved by ({u:.2f}, {v:.2f})")

    # Visualize markerless results.
    plt.figure(figsize=(6, 6))
    plt.imshow(tracker_ml.to_gray(I1), cmap='gray')
    for (x, y), (u, v) in flows_ml:
        plt.plot(x, y, 'go')
        plt.arrow(x, y, u, v, color='r', head_width=1.0, head_length=1.0)
    plt.title("Parallelized & Colored Input Lucas-Kanade Flow")
    plt.show()