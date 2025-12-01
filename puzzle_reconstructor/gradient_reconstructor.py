import cv2
import numpy as np
import os
import sys

# Asegurar que se puede importar puzzle_base desde el mismo directorio
sys.path.insert(0, os.path.dirname(__file__))

from puzzle_base import PuzzleSolverBase

class GradientSolver(PuzzleSolverBase):
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = ""):
        super().__init__(sliced_dir, output_dir, image_name)
    
    def extract_features(self, img: np.ndarray):
        """
        Calcula la magnitud del gradiente (bordes/líneas) en la zona de contacto.
        Ayuda a conectar líneas o contornos que cruzan entre piezas.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Sobel para detectar cambios de intensidad (bordes)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        
        w_b = self.border_width
        return {
            'top': magnitude[0:w_b, :],
            'bottom': magnitude[-w_b:, :],
            'left': magnitude[:, 0:w_b],
            'right': magnitude[:, -w_b:]
        }

    def calculate_cost(self, idx_a: int, idx_b: int, direction: str) -> float:
        """
        Compara los píxeles de gradiente en la frontera.
        Si las líneas continúan, la diferencia de gradiente debe ser baja.
        """
        feats_a = self.slices[idx_a].borders
        feats_b = self.slices[idx_b].borders
        
        if direction == 'horizontal':
            # Borde derecho de A (última columna) vs Izquierdo de B (primera columna)
            edge_a = feats_a['right'][:, -1]
            edge_b = feats_b['left'][:, 0]
        else: # vertical
            # Borde inferior de A (última fila) vs Superior de B (primera fila)
            edge_a = feats_a['bottom'][-1, :]
            edge_b = feats_b['top'][0, :]
            
        return np.mean(np.abs(edge_a - edge_b))

if __name__ == "__main__":
    import sys
    
    # Usar argumento de línea de comandos o pedir al usuario
    if len(sys.argv) > 1:
        NOMBRE_BASE = sys.argv[1]
    else:
        NOMBRE_BASE = input("Nombre base de la imagen (sin _slice_XXX.png): ").strip()
    
    solver = GradientSolver("sliced_images", "output_images", NOMBRE_BASE)
    try:
        print(f"Reconstruyendo '{NOMBRE_BASE}' con método GRADIENTE...")
        solver.load_slices(NOMBRE_BASE)
        solver.solve()
    except Exception as e:
        print(f"Error: {e}")
