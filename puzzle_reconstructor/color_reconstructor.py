import cv2
import numpy as np
import os
import sys

# Asegurar que se puede importar puzzle_base desde el mismo directorio
sys.path.insert(0, os.path.dirname(__file__))

from puzzle_base import PuzzleSolverBase

class ColorSolver(PuzzleSolverBase):
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = ""):
        super().__init__(sliced_dir, output_dir, image_name)
    
    def extract_features(self, img: np.ndarray):
        """
        Convierte a espacio de color LAB.
        L = Luminosidad, a/b = canales de color.
        Es mucho mejor que RGB para comparar similitud visual.
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
        """Calcula la distancia de color entre los bordes de contacto."""
        feats_a = self.slices[idx_a].borders
        feats_b = self.slices[idx_b].borders
        
        if direction == 'horizontal':
            edge_a = feats_a['right'][:, -1]
            edge_b = feats_b['left'][:, 0]
        else: # vertical
            edge_a = feats_a['bottom'][-1, :]
            edge_b = feats_b['top'][0, :]
            
        # Distancia Euclidiana entre los vectores de color (L, a, b)
        # axis=1 porque shape es (N, 3)
        diff = np.linalg.norm(edge_a - edge_b, axis=1)
        return np.mean(diff)

if __name__ == "__main__":
    import sys
    
    # Usar argumento de línea de comandos o pedir al usuario
    if len(sys.argv) > 1:
        NOMBRE_BASE = sys.argv[1]
    else:
        NOMBRE_BASE = input("Nombre base de la imagen (sin _slice_XXX.png): ").strip()
    
    solver = ColorSolver("sliced_images", "output_images", NOMBRE_BASE)
    try:
        print(f"Reconstruyendo '{NOMBRE_BASE}' con método COLOR...")
        solver.load_slices(NOMBRE_BASE)
        solver.solve()
    except Exception as e:
        print(f"Error: {e}")
