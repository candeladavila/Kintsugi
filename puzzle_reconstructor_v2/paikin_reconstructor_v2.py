"""
Paikin Reconstructor V2 - With Rotation Support

Advanced puzzle reconstruction using the Best Buddies algorithm
with rotation support (0°, 90°, 180°, 270°).
"""

import cv2
import numpy as np
import os
import sys
import math
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from puzzle_base_v2 import PuzzleSolverBaseV2, ImageSliceV2, rotate_image, ROTATIONS


class PaikinSolverV2(PuzzleSolverBaseV2):
    """
    Paikin solver with rotation support.

    Uses LAB color space and gradient magnitude for feature extraction,
    with dynamic weighting based on content type.
    """

    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = "", border_width: int = 30):
        super().__init__(sliced_dir, output_dir, image_name, border_width)
        self._cost_cache = {}

        # Parameters
        self.feature_border_width = 4
        self.multi_border_width = 3

        # Background activity threshold
        self.bg_activity_threshold = 0.03

        # Base weights (dynamically adjusted)
        self.weight_color = 0.4
        self.weight_gradient = 0.6

    def extract_features_rotated(self, img: np.ndarray, rotation: int) -> Dict[str, np.ndarray]:
        """
        Extracts LAB + gradient features from a rotated version of the image.

        Returns a dict with borders ('top', 'bottom', 'left', 'right') from a 4-channel feature image:
        - 3 channels: LAB (normalized to [0,1])
        - 1 channel: gradient magnitude (normalized to [0,1])
        """
        rotated = rotate_image(img, rotation)
        h, w = rotated.shape[:2]

        # LAB color space normalized to [0,1]
        lab = cv2.cvtColor(rotated, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0

        # Gradient magnitude (Scharr) normalized to [0,1]
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        g_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        grad_mag = cv2.magnitude(g_x, g_y)
        grad_mag = np.clip(grad_mag, 0, 1.0)

        # Combine: 3 LAB channels + 1 magnitude channel = 4 channels
        combined = np.dstack([lab, grad_mag[..., None]]).astype(np.float32)

        w_b = min(self.feature_border_width, h // 2, w // 2)
        return {
            "top": combined[0:w_b, :, :],
            "bottom": combined[-w_b:, :, :],
            "left": combined[:, 0:w_b, :],
            "right": combined[:, -w_b:, :],
        }

    def calculate_cost(
        self,
        idx_a: int,
        idx_b: int,
        direction: str,
        rotation_a: int = 0,
        rotation_b: int = 0,
    ) -> float:
        """
        Calculates compatibility cost between two pieces, considering rotations.

        Uses dynamic weighting based on whether the borders look like background or object content.

        Args:
            idx_a: Index of piece A
            idx_b: Index of piece B
            direction: 'horizontal' (A left of B) or 'vertical' (A above B)
            rotation_a: Rotation applied to piece A
            rotation_b: Rotation applied to piece B

        Returns:
            Cost value (lower = better match)
        """
        cache_key = (idx_a, idx_b, direction, rotation_a, rotation_b)
        if cache_key in self._cost_cache:
            return self._cost_cache[cache_key]

        feats_a = self.extract_features_rotated(self.slices[idx_a].original_image, rotation_a)
        feats_b = self.extract_features_rotated(self.slices[idx_b].original_image, rotation_b)

        mbw = self.multi_border_width

        if direction == "horizontal":
            # A is left of B: compare right border of A with left border of B
            max_w = min(feats_a["right"].shape[1], feats_b["left"].shape[1], mbw)
            edge_a = feats_a["right"][:, -max_w:, :]
            edge_b = feats_b["left"][:, :max_w, :]
        else:  # vertical
            # A is above B: compare bottom border of A with top border of B
            max_h = min(feats_a["bottom"].shape[0], feats_b["top"].shape[0], mbw)
            edge_a = feats_a["bottom"][-max_h:, :, :]
            edge_b = feats_b["top"][:max_h, :, :]

        # Flatten to [N pixels, 4 channels]
        edge_a_flat = edge_a.reshape(-1, edge_a.shape[-1])
        edge_b_flat = edge_b.reshape(-1, edge_b.shape[-1])

        # Measure "activity" using the magnitude channel (index 3)
        act_a = float(np.mean(edge_a_flat[:, 3]))
        act_b = float(np.mean(edge_b_flat[:, 3]))

        is_bg_a = act_a < self.bg_activity_threshold
        is_bg_b = act_b < self.bg_activity_threshold

        # Base differences
        diff_color = float(np.linalg.norm(edge_a_flat[:, :3] - edge_b_flat[:, :3], axis=1).mean())
        diff_grad = float(np.abs(edge_a_flat[:, 3] - edge_b_flat[:, 3]).mean())

        # Dynamic weighting based on content type
        if is_bg_a and is_bg_b:
            # Both likely background: color dominates
            cost = diff_color * 2.0
        elif is_bg_a != is_bg_b:
            # Mixed background/object: high penalty
            cost = 5.0 + diff_color + diff_grad
        else:
            # Both likely object edges: weighted combination
            cost = (self.weight_color * diff_color) + (self.weight_gradient * diff_grad)

        self._cost_cache[cache_key] = float(cost)
        return float(cost)

    def find_top_left_corner(self, n_slices: int) -> Tuple[int, int]:
        """
        Heuristic method: finds the piece and rotation that is probably the top-left corner.

        NOTE: If you always fix the top-left corner to '*_slice_000.png', you can remove this method.
        """
        max_min_cost = -1
        best_candidate = 0
        best_rotation = 0

        for i in range(n_slices):
            for rotation in ROTATIONS:
                costs_left = []
                costs_top = []

                for j in range(n_slices):
                    if i != j:
                        costs_left.append(self.calculate_cost(j, i, "horizontal", 0, rotation))
                        costs_top.append(self.calculate_cost(j, i, "vertical", 0, rotation))

                min_left = min(costs_left) if costs_left else 0
                min_top = min(costs_top) if costs_top else 0

                corner_score = min_left + min_top

                if corner_score > max_min_cost:
                    max_min_cost = corner_score
                    best_candidate = i
                    best_rotation = rotation

        return best_candidate, best_rotation

    def solve(self):
        """
        Greedy algorithm to reconstruct the puzzle with rotation support.

        IMPORTANT:
        - The top-left corner is FIXED to the piece '*_slice_000.png' with rotation 0°.
          This removes the global 0/90/180/270 ambiguity in the final assembled image.
        """
        n_slices = len(self.slices)
        side = int(math.sqrt(n_slices))
        rows, cols = side, side

        print(f"\n{'='*60}")
        print("PAIKIN RECONSTRUCTOR V2 - With Rotation Support")
        print(f"{'='*60}")
        print(f"Pieces: {n_slices} ({rows}x{cols})")
        print("Features: LAB + Gradient Magnitude")
        print("Rotations: 0°, 90°, 180°, 270°")
        print(f"{'='*60}\n")

        grid = [[None for _ in range(cols)] for _ in range(rows)]
        used_indices = set()

        # 1) Fixed top-left corner (no heuristic / no fallback)
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
        used_indices.add(start_idx)

        # 2) Fill the grid
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0:
                    continue

                best_idx = -1
                best_rotation = 0
                min_cost = float("inf")

                for idx in range(n_slices):
                    if idx in used_indices:
                        continue

                    for rotation in ROTATIONS:
                        cost = 0.0
                        count = 0

                        # Compare with left neighbor
                        if c > 0:
                            left_slice = grid[r][c - 1]
                            left_idx = self.slices.index(left_slice)  # robust: avoid relying on piece.id
                            cost += self.calculate_cost(
                                left_idx,
                                idx,
                                "horizontal",
                                left_slice.current_rotation,
                                rotation,
                            )
                            count += 1

                        # Compare with top neighbor
                        if r > 0:
                            top_slice = grid[r - 1][c]
                            top_idx = self.slices.index(top_slice)  # robust: avoid relying on piece.id
                            cost += self.calculate_cost(
                                top_idx,
                                idx,
                                "vertical",
                                top_slice.current_rotation,
                                rotation,
                            )
                            count += 1

                        avg_cost = cost / count if count > 0 else float("inf")

                        if avg_cost < min_cost:
                            min_cost = avg_cost
                            best_idx = idx
                            best_rotation = rotation

                # Safety fallback: place any remaining piece (rotation 0)
                if best_idx == -1:
                    best_idx = next(i for i in range(n_slices) if i not in used_indices)
                    best_rotation = 0

                self.slices[best_idx].set_rotation(best_rotation)
                grid[r][c] = self.slices[best_idx]
                used_indices.add(best_idx)

                progress = len(used_indices) / n_slices * 100
                print(
                    f"  [{progress:5.1f}%] Position ({r},{c}): piece #{best_idx} "
                    f"rot {best_rotation}° (cost: {min_cost:.2f})"
                )

        print("\n✓ Reconstruction completed\n")
        self.save_results(grid, rows, cols)

        print(f"{'='*60}")
        print("✓ Process completed")
        print(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Paikin Reconstructor V2 - With Rotation Support")
    parser.add_argument("image_name", nargs="?", help="Base name of the image")
    parser.add_argument("--sliced-dir", default="sliced_images_v2", help="Directory with sliced pieces")
    parser.add_argument("--output", "-o", default="output_images_v2", help="Output directory")

    args = parser.parse_args()

    if not args.image_name:
        args.image_name = input("Base image name (without _slice_XXX.png): ").strip()

    solver = PaikinSolverV2(args.sliced_dir, args.output, args.image_name)
    try:
        print(f"Reconstructing '{args.image_name}' with PAIKIN V2 method...")
        solver.load_slices(args.image_name)
        solver.solve()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
