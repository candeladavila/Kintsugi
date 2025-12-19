import cv2
import numpy as np
import os
import sys

# Ensure puzzle_base can be imported from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from puzzle_base import PuzzleSolverBase

class ColorSolver(PuzzleSolverBase):
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = "", border_width: int = 100):
        super().__init__(sliced_dir, output_dir, image_name, border_width)
    
    def extract_features(self, img: np.ndarray):
        """
        Convert to LAB color space.
        L = lightness, a/b = color channels.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        w_b = self.border_width
        
        return {
            'top': lab[0:w_b, :, :],
            'bottom': lab[-w_b:, :, :],
            'left': lab[:, 0:w_b, :],
            'right': lab[:, -w_b:, :]
        }

    def calculate_cost(self, idx_a: int, idx_b: int, direction: str) -> float:
        """Calculates the color distance between contacting edges."""
        feats_a = self.slices[idx_a].borders
        feats_b = self.slices[idx_b].borders
        
        if direction == 'horizontal':
            # Extract right edge of A and left edge of B
            # Borders are (height, width, 3), so we take the last column and first column
            edge_a = feats_a['right'][:, -1, :]  # (height, 3)
            edge_b = feats_b['left'][:, 0, :]     # (height, 3)
        else: # vertical
            # Extract bottom edge of A and top edge of B
            edge_a = feats_a['bottom'][-1, :, :]  # (width, 3)
            edge_b = feats_b['top'][0, :, :]      # (width, 3)
            
        # Euclidean distance between the color vectors (L, a, b)
        # axis=1 because shape is (N, 3) where N is height or width
        diff = np.linalg.norm(edge_a - edge_b, axis=1)
        return np.mean(diff)

if __name__ == "__main__":
    import sys
    
    # Use command line argument or ask user
    if len(sys.argv) > 1:
        NOMBRE_BASE = sys.argv[1]
    else:
        NOMBRE_BASE = input("Nombre base de la imagen (sin _slice_XXX.png): ").strip()
    
    solver = ColorSolver("sliced_images", "output_images", NOMBRE_BASE)
    try:
        print(f"Reconstructing '{NOMBRE_BASE}' with COLOR method...")
        solver.load_slices(NOMBRE_BASE)
        solver.solve()
    except Exception as e:
        print(f"Error: {e}")
