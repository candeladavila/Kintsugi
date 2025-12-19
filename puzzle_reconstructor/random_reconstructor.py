import math
import os
import sys

# Ensure puzzle_base can be imported from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from puzzle_base import PuzzleSolverBase

class RandomSolver(PuzzleSolverBase):
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = "", border_width: int = 10):
        super().__init__(sliced_dir, output_dir, image_name, border_width)
    
    def solve(self):
        """
        Override the solve method to NOT sort anything.
        Simply place pieces in the order they were read from disk.
        Since the input is shuffled, the output will display that order.
        """
        n_slices = len(self.slices)
        side = int(math.sqrt(n_slices))
        
        grid = []
        iterator = iter(self.slices)
        
        print("Generating random view (read order)...")
        
        for r in range(side):
            row = []
            for c in range(side):
                try:
                    row.append(next(iterator))
                except StopIteration:
                    break
            grid.append(row)
            
        self.save_results(grid, side, side)

if __name__ == "__main__":
    import sys
    
    # Use command line argument or ask user
    if len(sys.argv) > 1:
        NOMBRE_BASE = sys.argv[1]
    else:
        NOMBRE_BASE = input("Base image name (without _slice_XXX.png): ").strip()
    
    solver = RandomSolver("sliced_images", "output_images", NOMBRE_BASE)
    try:
        print(f"Showing '{NOMBRE_BASE}' in RANDOM order...")
        solver.load_slices(NOMBRE_BASE)
        solver.solve()
    except Exception as e:
        print(f"Error: {e}")
