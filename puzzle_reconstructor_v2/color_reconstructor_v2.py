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
    
    Uses LAB color space for perceptually accurate color matching
    while testing all possible rotation combinations.
    """
    
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = "", border_width: int = 10):
        super().__init__(sliced_dir, output_dir, image_name, border_width)
        self._cost_cache = {}
    
    def extract_features_rotated(self, img: np.ndarray, rotation: int) -> Dict[str, np.ndarray]:
        """
        Extracts LAB color features from a rotated version of the image.
        """
        rotated = rotate_image(img, rotation)
        lab = cv2.cvtColor(rotated, cv2.COLOR_BGR2LAB).astype(np.float32)
        w_b = self.border_width
        
        return {
            'top': lab[0:w_b, :, :],
            'bottom': lab[-w_b:, :, :],
            'left': lab[:, 0:w_b, :],
            'right': lab[:, -w_b:, :]
        }

    def calculate_cost(self, idx_a: int, idx_b: int, direction: str,
                       rotation_a: int = 0, rotation_b: int = 0) -> float:
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
        
        # Extract features with rotations
        feats_a = self.extract_features_rotated(self.slices[idx_a].original_image, rotation_a)
        feats_b = self.extract_features_rotated(self.slices[idx_b].original_image, rotation_b)
        
        if direction == 'horizontal':
            edge_a = feats_a['right'][:, -1]  # Right edge of A
            edge_b = feats_b['left'][:, 0]    # Left edge of B
        else:  # vertical
            edge_a = feats_a['bottom'][-1, :]  # Bottom edge of A
            edge_b = feats_b['top'][0, :]      # Top edge of B
        
        # Euclidean distance between LAB color vectors
        diff = np.linalg.norm(edge_a - edge_b, axis=1)
        cost = np.mean(diff)
        
        self._cost_cache[cache_key] = cost
        return cost

    def find_top_left_corner(self, n_slices: int) -> Tuple[int, int]:
        """
        Finds the piece and rotation that is probably the top-left corner.
        Uses color uniformity at top and left edges as indicator.
        """
        max_min_cost = -1
        best_candidate = 0
        best_rotation = 0

        for i in range(n_slices):
            for rotation in ROTATIONS:
                # Calculate minimum costs from left and top with this rotation
                costs_left = []
                costs_top = []
                
                for j in range(n_slices):
                    if i != j:
                        costs_left.append(self.calculate_cost(j, i, 'horizontal', 0, rotation))
                        costs_top.append(self.calculate_cost(j, i, 'vertical', 0, rotation))
                
                min_left = min(costs_left) if costs_left else 0
                min_top = min(costs_top) if costs_top else 0
                
                # Higher score = worse match = more likely to be corner
                corner_score = min_left + min_top
                
                if corner_score > max_min_cost:
                    max_min_cost = corner_score
                    best_candidate = i
                    best_rotation = rotation
        
        return best_candidate, best_rotation

    def solve(self):
        """
        Greedy algorithm to reconstruct the puzzle with rotation support.
        """
        n_slices = len(self.slices)
        side = int(math.sqrt(n_slices))
        rows, cols = side, side
        
        print(f"\n{'='*60}")
        print(f"COLOR RECONSTRUCTOR V2 - With Rotation Support")
        print(f"{'='*60}")
        print(f"Pieces: {n_slices} ({rows}x{cols})")
        print(f"Color space: LAB")
        print(f"Rotations: 0°, 90°, 180°, 270°")
        print(f"{'='*60}\n")
        
        grid = [[None for _ in range(cols)] for _ in range(rows)]
        used_indices = set()
        
        # 1. Find corner
        print("Searching for top-left corner (with rotation)...")
        start_idx = next((i for i, slc in enumerate(self.slices)
                        if slc.filename.endswith("_slice_000.png")), None)

        if start_idx is None:
            start_idx, start_rotation = self.find_top_left_corner(n_slices)
        else:
            start_rotation = 0

        self.slices[start_idx].set_rotation(start_rotation)
        grid[0][0] = self.slices[start_idx]
        used_indices.add(start_idx)
        
        # 2. Fill grid
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0:
                    continue
                
                best_idx = -1
                best_rotation = 0
                min_cost = float('inf')
                
                for idx in range(n_slices):
                    if idx in used_indices:
                        continue
                    
                    # Try all rotations
                    for rotation in ROTATIONS:
                        cost = 0
                        count = 0
                        
                        # Compare with left neighbor
                        if c > 0:
                            left_slice = grid[r][c-1]
                            cost += self.calculate_cost(
                                left_slice.id, idx, 'horizontal',
                                left_slice.current_rotation, rotation
                            )
                            count += 1
                        
                        # Compare with top neighbor
                        if r > 0:
                            top_slice = grid[r-1][c]
                            cost += self.calculate_cost(
                                top_slice.id, idx, 'vertical',
                                top_slice.current_rotation, rotation
                            )
                            count += 1
                        
                        avg_cost = cost / count if count > 0 else float('inf')
                        
                        if avg_cost < min_cost:
                            min_cost = avg_cost
                            best_idx = idx
                            best_rotation = rotation
                
                # Safety fallback
                if best_idx == -1:
                    best_idx = next(i for i in range(n_slices) if i not in used_indices)
                    best_rotation = 0
                
                self.slices[best_idx].set_rotation(best_rotation)
                grid[r][c] = self.slices[best_idx]
                used_indices.add(best_idx)
                
                progress = len(used_indices) / n_slices * 100
                print(f"  [{progress:5.1f}%] Position ({r},{c}): piece #{best_idx} rot {best_rotation}° (cost: {min_cost:.2f})")
        
        print(f"\n✓ Reconstruction completed\n")
        self.save_results(grid, rows, cols)
        
        print(f"{'='*60}")
        print(f"✓ Process completed")
        print(f"{'='*60}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Color Reconstructor V2 - With Rotation Support')
    parser.add_argument('image_name', nargs='?', help='Base name of the image')
    parser.add_argument('--sliced-dir', default='sliced_images_v2', help='Directory with sliced pieces')
    parser.add_argument('--output', '-o', default='output_images_v2', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.image_name:
        args.image_name = input("Base image name (without _slice_XXX.png): ").strip()
    
    solver = ColorSolverV2(args.sliced_dir, args.output, args.image_name)
    try:
        print(f"Reconstructing '{args.image_name}' with COLOR V2 method...")
        solver.load_slices(args.image_name)
        solver.solve()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
