"""
Random Reconstructor V2 - With Rotation Support

Baseline reconstructor that shows pieces in their original
random order and rotation (no solving algorithm).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from puzzle_base_v2 import PuzzleSolverBaseV2


class RandomSolverV2(PuzzleSolverBaseV2):
    """
    Random solver for V2 (with rotation).
    
    Simply places pieces in the order they were read from disk,
    with a random rotation applied to each. This serves as a
    baseline to compare against other algorithms.
    """
    
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = "", border_width: int = 10):
        super().__init__(sliced_dir, output_dir, image_name, border_width)
    
    def calculate_cost(self, idx_a: int, idx_b: int, direction: str,
                       rotation_a: int = 0, rotation_b: int = 0) -> float:
        """Returns 0 as we don't calculate costs for random placement."""
        return 0.0
    
    def solve(self):
        """
        Overrides the solve method to NOT sort anything.
        Places pieces in reading order without any optimization.
        """
        n_slices = len(self.slices)
        side = int(math.sqrt(n_slices))
        
        grid = []
        iterator = iter(self.slices)
        
        print(f"\n{'='*60}")
        print(f"RANDOM RECONSTRUCTOR V2 - Baseline (No Solving)")
        print(f"{'='*60}")
        print(f"Pieces: {n_slices} ({side}x{side})")
        print(f"Mode: Random order + Random rotation (as loaded)")
        print(f"{'='*60}\n")
        
        print("Generating random view (reading order)...")
        
        for r in range(side):
            row = []
            for c in range(side):
                try:
                    piece = next(iterator)
                    # Keep rotation at 0 (as loaded) for V2
                    # The pieces were already rotated during slicing
                    piece.set_rotation(0)
                    row.append(piece)
                except StopIteration:
                    break
            grid.append(row)
        
        print(f"✓ {n_slices} pieces placed in random order\n")
        self.save_results(grid, side, side)
        
        print(f"{'='*60}")
        print(f"✓ Process completed (baseline - no optimization)")
        print(f"{'='*60}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Random Reconstructor V2 - Baseline')
    parser.add_argument('image_name', nargs='?', help='Base name of the image')
    parser.add_argument('--sliced-dir', default='sliced_images_v2', help='Directory with sliced pieces')
    parser.add_argument('--output', '-o', default='output_images_v2', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.image_name:
        args.image_name = input("Base image name (without _slice_XXX.png): ").strip()
    
    solver = RandomSolverV2(args.sliced_dir, args.output, args.image_name)
    try:
        print(f"Showing '{args.image_name}' in RANDOM order (V2)...")
        solver.load_slices(args.image_name)
        solver.solve()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
