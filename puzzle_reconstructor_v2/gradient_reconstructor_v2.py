"""
Gradient Reconstructor V2 - With Rotation Support

Enhanced gradient-based puzzle reconstruction that considers
piece rotations (0°, 90°, 180°, 270°).
"""

import cv2
import numpy as np
import os
import sys
import math
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
from puzzle_base_v2 import PuzzleSolverBaseV2, ImageSliceV2, rotate_image, ROTATIONS


class GradientSolverV2(PuzzleSolverBaseV2):
    """
    Gradient-based solver with rotation support.

    Method overview:
    1) Extracts borders from each piece (configurable width)
    2) For each pair, tests all rotation combinations
    3) Analyzes gradients in LAB color space
    4) Assigns a quality score based on gradient and color continuity
    5) Performs greedy placement using the best local matches
    """

    def __init__(
        self,
        sliced_dir: str,
        output_dir: str,
        image_name: str = "",
        border_width: int = 10,
    ):
        super().__init__(sliced_dir, output_dir, image_name, border_width)

        # NOTE: This overrides the incoming `border_width` argument to a fixed value.
        # If you want it configurable, replace `100` with `border_width`.
        self.border_width = 100  # Border width to extract (pixels)

        # Compatibility cache between pieces and rotations
        # Structure: {(idx_a, rot_a, idx_b, rot_b, direction): quality}
        self.compatibility_cache: Dict[tuple, float] = {}

    def extract_border(self, img: np.ndarray, side: str) -> np.ndarray:
        """
        Extracts a border from the image.

        Args:
            img: Image to extract border from
            side: 'top', 'bottom', 'left', 'right'

        Returns:
            Extracted border region
        """
        h, w = img.shape[:2]
        border_w = min(self.border_width, w, h)

        if side == "top":
            return img[0:border_w, :].copy()
        elif side == "bottom":
            return img[-border_w:, :].copy()
        elif side == "left":
            return img[:, 0:border_w].copy()
        elif side == "right":
            return img[:, -border_w:].copy()
        else:
            raise ValueError(f"Invalid side: {side}")

    def extract_border_rotated(self, img: np.ndarray, side: str, rotation: int) -> np.ndarray:
        """
        Extracts a border from a rotated version of the image.

        Args:
            img: Original image
            side: 'top', 'bottom', 'left', 'right'
            rotation: Rotation to apply first

        Returns:
            Border extracted from the rotated image
        """
        rotated = rotate_image(img, rotation)
        return self.extract_border(rotated, side)

    def create_border_pair_image(self, border1: np.ndarray, border2: np.ndarray, orientation: str) -> np.ndarray:
        """
        Combines two borders into a single image for continuity analysis.

        Args:
            border1: First border image
            border2: Second border image
            orientation: 'horizontal' or 'vertical'

        Returns:
            Combined border image
        """
        if orientation == "horizontal":
            return np.hstack([border1, border2])
        elif orientation == "vertical":
            return np.vstack([border1, border2])
        else:
            raise ValueError(f"Invalid orientation: {orientation}")

    def calculate_gradient_continuity(self, combined_border: np.ndarray, orientation: str) -> float:
        """
        Computes a continuity quality score at the junction of two borders.
        Lower is better.

        Uses LAB color space to better approximate perceptual differences.

        Args:
            combined_border: Concatenated border image (border A + border B)
            orientation: 'horizontal' or 'vertical'

        Returns:
            Quality score (lower = better continuity)
        """
        # Convert to LAB for perceptual analysis
        lab = cv2.cvtColor(combined_border, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Compute gradient magnitude for each LAB channel
        gradients_lab = []
        for channel in range(3):
            grad_x = cv2.Sobel(lab[:, :, channel], cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(lab[:, :, channel], cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            gradients_lab.append(grad_mag)

        # Weighted combined gradient magnitude (L has higher weight)
        grad_magnitude = (
            gradients_lab[0] * 0.6
            + gradients_lab[1] * 0.2
            + gradients_lab[2] * 0.2
        )

        h, w = combined_border.shape[:2]

        if orientation == "horizontal":
            # Junction is at the vertical center line between both borders
            center = w // 2
            region_width = 30
            left_bound = max(0, center - region_width // 2)
            right_bound = min(w, center + region_width // 2)

            junction_region_grad = grad_magnitude[:, left_bound:right_bound]
            junction_region_lab = lab[:, left_bound:right_bound, :]

            center_local = (right_bound - left_bound) // 2
            if 0 < center_local < junction_region_grad.shape[1] - 1:
                left_grad = junction_region_grad[:, center_local - 1]
                right_grad = junction_region_grad[:, center_local + 1]
                diff_grad = np.mean(np.abs(left_grad - right_grad))

                left_color = junction_region_lab[:, center_local - 1, :]
                right_color = junction_region_lab[:, center_local + 1, :]
                diff_color = np.mean(np.sqrt(np.sum((left_color - right_color) ** 2, axis=1)))
            else:
                diff_grad = np.mean(junction_region_grad)
                diff_color = 100.0

        elif orientation == "vertical":
            # Junction is at the horizontal center line between both borders
            center = h // 2
            region_height = 30
            top_bound = max(0, center - region_height // 2)
            bottom_bound = min(h, center + region_height // 2)

            junction_region_grad = grad_magnitude[top_bound:bottom_bound, :]
            junction_region_lab = lab[top_bound:bottom_bound, :, :]

            center_local = (bottom_bound - top_bound) // 2
            if 0 < center_local < junction_region_grad.shape[0] - 1:
                top_grad = junction_region_grad[center_local - 1, :]
                bottom_grad = junction_region_grad[center_local + 1, :]
                diff_grad = np.mean(np.abs(top_grad - bottom_grad))

                top_color = junction_region_lab[center_local - 1, :, :]
                bottom_color = junction_region_lab[center_local + 1, :, :]
                diff_color = np.mean(np.sqrt(np.sum((top_color - bottom_color) ** 2, axis=1)))
            else:
                diff_grad = np.mean(junction_region_grad)
                diff_color = 100.0
        else:
            return float("inf")

        # Smoothness metrics around the junction region
        smoothness_grad = np.std(junction_region_grad)
        smoothness_color = np.std(junction_region_lab)

        # Combined quality score (lower = better)
        quality_index = (
            0.4 * diff_grad
            + 0.4 * diff_color
            + 0.1 * smoothness_grad
            + 0.1 * smoothness_color
        )
        return float(quality_index)

    def calculate_cost(
        self,
        idx_a: int,
        idx_b: int,
        direction: str,
        rotation_a: int = 0,
        rotation_b: int = 0,
    ) -> float:
        """
        Calculates compatibility (quality score) between two pieces, considering rotations.

        Args:
            idx_a: Index of piece A (in self.slices)
            idx_b: Index of piece B (in self.slices)
            direction: 'horizontal' (A left of B) or 'vertical' (A above B)
            rotation_a: Rotation applied to piece A
            rotation_b: Rotation applied to piece B

        Returns:
            Quality score (lower = better match)
        """
        img_a = self.slices[idx_a].original_image
        img_b = self.slices[idx_b].original_image

        if direction == "horizontal":
            # A is left of B: compare right border of A with left border of B
            border_a = self.extract_border_rotated(img_a, "right", rotation_a)
            border_b = self.extract_border_rotated(img_b, "left", rotation_b)
            orientation = "horizontal"
        elif direction == "vertical":
            # A is above B: compare bottom border of A with top border of B
            border_a = self.extract_border_rotated(img_a, "bottom", rotation_a)
            border_b = self.extract_border_rotated(img_b, "top", rotation_b)
            orientation = "vertical"
        else:
            raise ValueError(f"Invalid direction: {direction}")

        try:
            combined = self.create_border_pair_image(border_a, border_b, orientation)
            quality = self.calculate_gradient_continuity(combined, orientation)
        except Exception:
            quality = float("inf")

        return float(quality)

    def build_compatibility_matrix(self):
        """
        Builds a compatibility cache for all piece pairs and all rotation combinations.
        """
        n = len(self.slices)
        print(f"[{self.__class__.__name__}] Building compatibility cache with rotations...")
        print(f"  Analyzing {n} pieces with {self.border_width}px borders...")
        print(f"  Testing {len(ROTATIONS)} rotations per piece...")

        self.compatibility_cache = {}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                for rot_a in ROTATIONS:
                    for rot_b in ROTATIONS:
                        # Horizontal compatibility
                        q_h = self.calculate_cost(i, j, "horizontal", rot_a, rot_b)
                        self.compatibility_cache[(i, rot_a, j, rot_b, "horizontal")] = q_h

                        # Vertical compatibility
                        q_v = self.calculate_cost(i, j, "vertical", rot_a, rot_b)
                        self.compatibility_cache[(i, rot_a, j, rot_b, "vertical")] = q_v

            progress = ((i + 1) / n) * 100
            if (i + 1) % max(1, n // 10) == 0:
                print(f"  Progress: {progress:.1f}%")

        print("  ✓ Compatibility cache completed\n")

    def find_best_match(
        self,
        piece_idx: int,
        piece_rotation: int,
        direction: str,
        used_pieces: set,
    ) -> Tuple[int, int, float]:
        """
        Finds the best candidate piece (and its rotation) to match a given piece.

        Args:
            piece_idx: Index of the reference piece (in self.slices)
            piece_rotation: Rotation of the reference piece
            direction: 'horizontal' or 'vertical'
            used_pieces: Set of indices already placed

        Returns:
            Tuple (best_index, best_rotation, best_quality)
        """
        best_idx = -1
        best_rotation = 0
        best_quality = float("inf")

        for j in range(len(self.slices)):
            if j in used_pieces:
                continue

            for rot_b in ROTATIONS:
                key = (piece_idx, piece_rotation, j, rot_b, direction)
                quality = self.compatibility_cache.get(
                    key,
                    self.calculate_cost(piece_idx, j, direction, piece_rotation, rot_b),
                )

                if quality < best_quality:
                    best_quality = quality
                    best_idx = j
                    best_rotation = rot_b

        return best_idx, best_rotation, float(best_quality)

    def solve(self):
        """
        Main solving method with rotation support.

        IMPORTANT:
        - The top-left corner is FIXED to the piece '*_slice_000.png' with rotation 0°.
          This removes the global 0/90/180/270 ambiguity in the final assembled image.
        """
        n_slices = len(self.slices)
        side = int(math.sqrt(n_slices))
        rows, cols = side, side

        print(f"\n{'='*60}")
        print("GRADIENT RECONSTRUCTOR V2 - With Rotation Support")
        print(f"{'='*60}")
        print(f"Pieces: {n_slices} ({rows}x{cols})")
        print(f"Border width: {self.border_width}px")
        print("Rotations: 0°, 90°, 180°, 270°")
        print(f"{'='*60}\n")

        # Step 1: Build compatibility cache
        self.build_compatibility_matrix()

        # Step 2: Greedy reconstruction
        grid = [[None for _ in range(cols)] for _ in range(rows)]
        used_pieces = set()

        print(f"[{self.__class__.__name__}] Starting greedy reconstruction with rotations...")

        # 2.1) Fixed top-left corner (no heuristic / no fallback)
        start_idx = next(
            (i for i, slc in enumerate(self.slices) if slc.filename.endswith("_slice_000.png")),
            None,
        )
        if start_idx is None:
            raise ValueError(
                "No piece named '*_slice_000.png' was found. "
                "To always fix the top-left corner, make sure that piece exists."
            )

        start_rotation = 0
        self.slices[start_idx].set_rotation(start_rotation)
        grid[0][0] = self.slices[start_idx]
        used_pieces.add(start_idx)

        # 2.2) Fill the grid
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] is not None:
                    continue

                candidates = []

                # If there is a piece to the left
                if c > 0 and grid[r][c - 1] is not None:
                    left_piece = grid[r][c - 1]
                    left_idx = self.slices.index(left_piece)  # robust: avoid relying on piece.id
                    best_idx, best_rot, quality = self.find_best_match(
                        left_idx,
                        left_piece.current_rotation,
                        "horizontal",
                        used_pieces,
                    )
                    if best_idx >= 0:
                        candidates.append((best_idx, best_rot, quality, "h"))

                # If there is a piece above
                if r > 0 and grid[r - 1][c] is not None:
                    top_piece = grid[r - 1][c]
                    top_idx = self.slices.index(top_piece)  # robust: avoid relying on piece.id
                    best_idx, best_rot, quality = self.find_best_match(
                        top_idx,
                        top_piece.current_rotation,
                        "vertical",
                        used_pieces,
                    )
                    if best_idx >= 0:
                        candidates.append((best_idx, best_rot, quality, "v"))

                if candidates:
                    # Pick the best candidate (lowest quality score)
                    candidates.sort(key=lambda x: x[2])
                    best_idx, best_rot, best_quality, _ = candidates[0]

                    self.slices[best_idx].set_rotation(best_rot)
                    grid[r][c] = self.slices[best_idx]
                    used_pieces.add(best_idx)

                    progress = len(used_pieces) / len(self.slices) * 100
                    print(
                        f"  [{progress:5.1f}%] Position ({r},{c}): piece #{best_idx} "
                        f"rot {best_rot}° (quality: {best_quality:.2f})"
                    )
                else:
                    # Fallback: place any remaining piece (rotation 0)
                    for idx in range(len(self.slices)):
                        if idx not in used_pieces:
                            self.slices[idx].set_rotation(0)
                            grid[r][c] = self.slices[idx]
                            used_pieces.add(idx)
                            break

        print(f"\n  ✓ Reconstruction completed: {len(used_pieces)}/{len(self.slices)} pieces\n")

        # Step 3: Save results
        self.save_results(grid, rows, cols)

        print(f"{'='*60}")
        print("✓ Process completed")
        print(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gradient Reconstructor V2 - With Rotation Support")
    parser.add_argument("sliced_dir", help="Directory with sliced pieces")
    parser.add_argument("--output", "-o", default="output_images_v2", help="Output directory")
    parser.add_argument("--name", "-n", default="image", help="Base image name")

    args = parser.parse_args()

    solver = GradientSolverV2(args.sliced_dir, args.output, args.name)
    solver.load_slices(args.name)
    solver.solve()


if __name__ == "__main__":
    main()
