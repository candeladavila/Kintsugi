import math
import os
import sys

# Asegurar que se puede importar puzzle_base desde el mismo directorio
sys.path.insert(0, os.path.dirname(__file__))

from puzzle_base import PuzzleSolverBase

class RandomSolver(PuzzleSolverBase):
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = ""):
        super().__init__(sliced_dir, output_dir, image_name)
    
    def solve(self):
        """
        Sobrescribe el método solve para NO ordenar nada.
        Simplemente coloca las piezas en el orden en que se leyeron del disco.
        Como el input son trozos desordenados, el output mostrará ese desorden.
        """
        n_slices = len(self.slices)
        side = int(math.sqrt(n_slices))
        
        grid = []
        iterator = iter(self.slices)
        
        print("Generando vista aleatoria (orden de lectura)...")
        
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
    
    # Usar argumento de línea de comandos o pedir al usuario
    if len(sys.argv) > 1:
        NOMBRE_BASE = sys.argv[1]
    else:
        NOMBRE_BASE = input("Nombre base de la imagen (sin _slice_XXX.png): ").strip()
    
    solver = RandomSolver("sliced_images", "output_images", NOMBRE_BASE)
    try:
        print(f"Mostrando '{NOMBRE_BASE}' en orden ALEATORIO...")
        solver.load_slices(NOMBRE_BASE)
        solver.solve()
    except Exception as e:
        print(f"Error: {e}")
