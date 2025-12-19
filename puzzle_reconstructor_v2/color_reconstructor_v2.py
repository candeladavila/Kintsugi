"""
Color Reconstructor V2 - With Rotation Support

Color-based puzzle reconstruction using LAB color space
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


class ColorSolverV2(PuzzleSolverBaseV2):
    """
    Color-based solver with rotation support.

    Uses LAB color space for better perceptual similarity comparisons.
    """

    def __init__(
        self,
        sliced_dir: str,
        output_dir: str = "output_images",
        image_name: str = "",
        border_width: int = 10,
    ):
        # Support old single-argument usage by allowing output_dir to be omitted.
        super().__init__(sliced_dir, output_dir, image_name, border_width)
        self._lab_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._cost_cache: Dict[Tuple[int, int, str, int, int], float] = {}

        # Full compatibility cache:
        # {(idx_a, rot_a, idx_b, rot_b, direction): cost}
        self.compatibility_cache: Dict[Tuple[int, int, int, int, str], float] = {}

    def _get_lab(self, idx: int, rotation: int = 0) -> np.ndarray:
        """
        Returns the LAB version of the slice image, with rotation applied.
        Uses caching for performance.
        """
        cache_key = (idx, rotation)
        if cache_key in self._lab_cache:
            return self._lab_cache[cache_key]

        img = self.slices[idx].original_image
        img_rot = rotate_image(img, rotation)

        lab = cv2.cvtColor(img_rot, cv2.COLOR_BGR2LAB)
        self._lab_cache[cache_key] = lab
        return lab

    def calculate_cost(
        self,
        idx_a: int,
        idx_b: int,
        direction: str,
        rotation_a: int = 0,
        rotation_b: int = 0,
    ) -> float:
        """
        Calculates the color distance between contact borders of two pieces.

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

        lab_a = self._get_lab(idx_a, rotation_a)
        lab_b = self._get_lab(idx_b, rotation_b)

        if direction == "horizontal":
            # Right edge of A vs left edge of B
            edge_a = lab_a[:, -1, :].astype(np.float32)
            edge_b = lab_b[:, 0, :].astype(np.float32)
        else:
            # Bottom edge of A vs top edge of B
            edge_a = lab_a[-1, :, :].astype(np.float32)
            edge_b = lab_b[0, :, :].astype(np.float32)

        # Euclidean distance in LAB
        diff = np.linalg.norm(edge_a - edge_b, axis=1)
        cost = float(np.mean(diff))

        self._cost_cache[cache_key] = cost
        return cost

    def build_compatibility_matrix(self):
        """
        Builds a full compatibility cache between all pieces for all rotation combinations.

        Structure: {(idx_a, rot_a, idx_b, rot_b, direction): cost}
        """
        n = len(self.slices)
        print(f"[{self.__class__.__name__}] Building compatibility cache with rotations...")
        print(f"  Analyzing {n} pieces...")
        print(f"  Testing {len(ROTATIONS)} rotations per piece...")

        self.compatibility_cache = {}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                for rot_a in ROTATIONS:
                    for rot_b in ROTATIONS:
                        # Horizontal compatibility
                        c_h = self.calculate_cost(i, j, "horizontal", rot_a, rot_b)
                        self.compatibility_cache[(i, rot_a, j, rot_b, "horizontal")] = c_h

                        # Vertical compatibility
                        c_v = self.calculate_cost(i, j, "vertical", rot_a, rot_b)
                        self.compatibility_cache[(i, rot_a, j, rot_b, "vertical")] = c_v

            progress = ((i + 1) / n) * 100
            if (i + 1) % max(1, n // 10) == 0:
                print(f"  Progress: {progress:.1f}%")

        print("  ✓ Compatibility cache completed\n")

    def find_best_match(
        self,
        piece_idx: int,
        piece_rotation: int,
        direction: str,
        used_indices: set,
    ) -> Tuple[int, int, float]:
        """
        Finds the best piece and rotation to pair with a given piece (using the precomputed cache).

        Returns:
            Tuple (best_index, best_rotation, cost)
        """
        best_idx = -1
        best_rotation = 0
        best_cost = float("inf")
        n = len(self.slices)

        for j in range(n):
            if j in used_indices or j == piece_idx:
                continue

            for rot_b in ROTATIONS:
                key = (piece_idx, piece_rotation, j, rot_b, direction)
                c = self.compatibility_cache.get(key)
                if c is None:
                    # Safe fallback (should not happen if cache is built)
                    c = self.calculate_cost(piece_idx, j, direction, piece_rotation, rot_b)

                if c < best_cost:
                    best_cost = c
                    best_idx = j
                    best_rotation = rot_b

        return best_idx, best_rotation, float(best_cost)

    def solve(self):
        """
        Main solving method with rotation support.
        """
        n_slices = len(self.slices)
        side = int(math.sqrt(n_slices))
        rows, cols = side, side

        print(f"\n{'='*60}")
        print("COLOR RECONSTRUCTOR V2 - With Rotation Support")
        print(f"{'='*60}")
        print(f"Pieces: {n_slices} ({rows}x{cols})")
        print("Color space: LAB")
        print("Rotations: 0°, 90°, 180°, 270°")
        print(f"{'='*60}\n")

        # Step 1: Build compatibility cache
        self.build_compatibility_matrix()

        # Step 2: Greedy reconstruction
        grid = [[None for _ in range(cols)] for _ in range(rows)]
        used_indices = set()

        print(f"[{self.__class__.__name__}] Starting greedy reconstruction with rotations...")

        # Fixed top-left corner (no heuristic / no fallback)
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

        # Build grid
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] is not None:
                    continue

                candidates = []

                # If there's a piece to the left
                if c > 0 and grid[r][c - 1] is not None:
                    left_piece = grid[r][c - 1]
                    left_idx = self.slices.index(left_piece)
                    best_idx, best_rot, cost = self.find_best_match(
                        left_idx,
                        left_piece.current_rotation,
                        "horizontal",
                        used_indices,
                    )
                    if best_idx >= 0:
                        candidates.append((best_idx, best_rot, cost, "h"))

                # If there's a piece above
                if r > 0 and grid[r - 1][c] is not None:
                    top_piece = grid[r - 1][c]
                    top_idx = self.slices.index(top_piece)
                    best_idx, best_rot, cost = self.find_best_match(
                        top_idx,
                        top_piece.current_rotation,
                        "vertical",
                        used_indices,
                    )
                    if best_idx >= 0:
                        candidates.append((best_idx, best_rot, cost, "v"))

                if candidates:
                    # Sort by cost and pick best
                    candidates.sort(key=lambda x: x[2])
                    best_idx, best_rot, best_cost, _ = candidates[0]

                    self.slices[best_idx].set_rotation(best_rot)
                    grid[r][c] = self.slices[best_idx]
                    used_indices.add(best_idx)

                    progress = len(used_indices) / len(self.slices) * 100
                    print(
                        f"  [{progress:5.1f}%] Position ({r},{c}): piece #{best_idx} "
                        f"rot {best_rot}° (cost: {best_cost:.2f})"
                    )
                else:
                    # Fallback: place any remaining piece (rotation 0)
                    for idx in range(len(self.slices)):
                        if idx not in used_indices:
                            self.slices[idx].set_rotation(0)
                            grid[r][c] = self.slices[idx]
                            used_indices.add(idx)
                            break

        print(f"\n  ✓ Reconstruction completed: {len(used_indices)}/{len(self.slices)} pieces\n")
        # Save results using base class helper
        try:
            self.save_results(grid, rows, cols)
        except Exception:
            # If saving fails, continue to return grid for debugging
            pass

        return grid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Color Reconstructor V2 - With Rotation')
    parser.add_argument('image_name', nargs='?', help='Base name of the image (without _slice_XXX.png)')
    parser.add_argument('--sliced-dir', default='sliced_images_v2', help='Directory with sliced pieces')
    parser.add_argument('--output', '-o', default='output_images/ver_2', help='Output directory')
    parser.add_argument('--border-width', type=int, default=100, help='Border width for analysis')

    args = parser.parse_args()

    if not args.image_name:
        args.image_name = input('Base image name (without _slice_XXX.png): ').strip()

    solver = ColorSolverV2(args.sliced_dir, args.output, args.image_name, args.border_width)
    try:
        print(f"Running color reconstructor V2 for '{args.image_name}'...")
        solver.load_slices(args.image_name)
        solver.solve()
    except Exception as e:
        print(f"Error: {e}")
