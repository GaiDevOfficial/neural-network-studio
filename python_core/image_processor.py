"""
Python Image Processor Pipeline
Handles 28x28 Resizing, Bounding Box Cropping, Center-of-Mass Centering,
Grayscale Luminance Conversion, and Tensor Normalization [0.0, 1.0].

File: python_core/image_processor.py
"""

import numpy as np


class ImageProcessor:
    @staticmethod
    def process_image_array(img_array, target_shape=(28, 28), threshold=0.05):
        """
        Process a 2D or 3D numpy image array into a normalized 28x28 flattened float array.
        Follows standard MNIST centering logic.
        """
        arr = np.array(img_array, dtype=float)

        # Convert RGB/RGBA to Grayscale if needed
        if arr.ndim == 3:
            if arr.shape[2] >= 3:
                # Standard luminance formula
                gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
                if arr.shape[2] == 4:
                    alpha = arr[:, :, 3] / 255.0 if arr[:, :, 3].max() > 1.0 else arr[:, :, 3]
                    gray = gray * alpha
            else:
                gray = arr[:, :, 0]
        else:
            gray = arr

        # Normalize to [0, 1] if values are in [0, 255]
        if gray.max() > 1.0:
            gray = gray / 255.0

        target_h, target_w = target_shape
        empty_tensor = np.zeros(target_h * target_w, dtype=float)

        # Detect active pixels & bounding box
        active_mask = gray > threshold
        if not np.any(active_mask):
            return {
                "tensor": empty_tensor,
                "is_empty": True,
                "shape": target_shape,
                "active_pixel_count": 0
            }

        rows = np.any(active_mask, axis=1)
        cols = np.any(active_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        cropped = gray[rmin:rmax + 1, cmin:cmax + 1]
        ch, cw = cropped.shape

        # Inner bounding box size (~20x20 in 28x28)
        inner_size = int(target_w * 0.72)
        if cw > ch:
            scale_w = inner_size
            scale_h = max(1, int(round((ch / cw) * inner_size)))
        else:
            scale_h = inner_size
            scale_w = max(1, int(round((cw / ch) * inner_size)))

        # Bilinear / Nearest interpolation for downsampling
        scaled = np.zeros((scale_h, scale_w), dtype=float)
        for r in range(scale_h):
            for c in range(scale_w):
                src_r = int((r / scale_h) * ch)
                src_c = int((c / scale_w) * cw)
                scaled[r, c] = cropped[src_r, src_c]

        # Centering canvas
        canvas = np.zeros((target_h, target_w), dtype=float)
        start_r = (target_h - scale_h) // 2
        start_c = (target_w - scale_w) // 2
        canvas[start_r:start_r + scale_h, start_c:start_c + scale_w] = scaled

        # Center-of-mass shift
        total_mass = np.sum(canvas)
        if total_mass > 1e-6:
            r_indices, c_indices = np.indices(canvas.shape)
            com_r = np.sum(r_indices * canvas) / total_mass
            com_c = np.sum(c_indices * canvas) / total_mass

            shift_r = int(np.clip(round(target_h / 2 - com_r), -4, 4))
            shift_c = int(np.clip(round(target_w / 2 - com_c), -4, 4))

            shifted = np.zeros_like(canvas)
            src_r_start = max(0, -shift_r)
            src_r_end = min(target_h, target_h - shift_r)
            dst_r_start = max(0, shift_r)
            dst_r_end = min(target_h, target_h + shift_r)

            src_c_start = max(0, -shift_c)
            src_c_end = min(target_w, target_w - shift_c)
            dst_c_start = max(0, shift_c)
            dst_c_end = min(target_w, target_w + shift_c)

            h_slice = min(src_r_end - src_r_start, dst_r_end - dst_r_start)
            w_slice = min(src_c_end - src_c_start, dst_c_end - dst_c_start)

            if h_slice > 0 and w_slice > 0:
                shifted[dst_r_start:dst_r_start + h_slice, dst_c_start:dst_c_start + w_slice] = \
                    canvas[src_r_start:src_r_start + h_slice, src_c_start:src_c_start + w_slice]
            canvas = shifted

        tensor = np.clip(canvas.flatten(), 0.0, 1.0)
        return {
            "tensor": tensor,
            "is_empty": False,
            "shape": target_shape,
            "active_pixel_count": int(np.sum(active_mask))
        }
