import os
import glob
import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class ImageSlice:
    id: int
    filename: str
    image: np.ndarray
    # Características (bordes) para el análisis
    borders: Dict[str, np.ndarray] 

class PuzzleSolverBase:
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = ""):
        self.sliced_dir = sliced_dir
        self.output_dir = output_dir
        self.image_name = image_name
        self.slices: List[ImageSlice] = []
        self.border_width = 10  # Ancho del borde a analizar (10 píxeles)

    def extract_features(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Extrae los bordes para el análisis. Puede ser sobrescrito."""
        w_b = self.border_width
        return {
            'top': img[0:w_b, :],
            'bottom': img[-w_b:, :],
            'left': img[:, 0:w_b],
            'right': img[:, -w_b:]
        }

    def load_slices(self, original_name_pattern: str):
        """Carga los trozos ya existentes en la carpeta especificada."""
        # Busca archivos con el patrón: nombre_slice_XXX.png en la carpeta específica
        search_pattern = os.path.join(self.sliced_dir, f"{original_name_pattern}_slice_*.png")
        
        # Ordenamos numéricamente para procesarlos en orden
        try:
            files = sorted(glob.glob(search_pattern), 
                          key=lambda x: int(x.split('_slice_')[1].split('.')[0]))
        except IndexError:
            # Fallback por si el nombre no sigue el formato exacto
            files = sorted(glob.glob(search_pattern))
        
        if not files:
            raise FileNotFoundError(f"No se encontraron imágenes en: {search_pattern}")
            
        print(f"[{self.__class__.__name__}] Cargando {len(files)} trozos desde {self.sliced_dir}...")
        
        for idx, fpath in enumerate(files):
            img = cv2.imread(fpath)
            if img is None: continue
            
            features = self.extract_features(img)
            self.slices.append(ImageSlice(idx, os.path.basename(fpath), img, features))

    def calculate_cost(self, idx_a: int, idx_b: int, direction: str) -> float:
        """Debe ser implementado por las subclases (Gradiente y Color)."""
        raise NotImplementedError

    def find_top_left_corner(self, n_slices: int) -> int:
        """
        Encuentra la pieza que tiene peor coincidencia arriba y a la izquierda.
        Esa pieza es probablemente la esquina superior izquierda.
        """
        max_min_cost = -1
        best_candidate = 0

        for i in range(n_slices):
            # Mejor coste posible a su izquierda (si fuera >0 seria malo)
            min_left = min([self.calculate_cost(j, i, 'horizontal') for j in range(n_slices) if i != j])
            # Mejor coste posible arriba
            min_top = min([self.calculate_cost(j, i, 'vertical') for j in range(n_slices) if i != j])
            
            # Sumamos los costes mínimos. Cuanto más alto sea este valor, 
            # menos se parece a ninguna otra pieza por esos lados.
            corner_score = min_left + min_top
            
            if corner_score > max_min_cost:
                max_min_cost = corner_score
                best_candidate = i
        
        return best_candidate

    def solve(self):
        """Algoritmo Greedy automático para reconstruir el puzzle."""
        n_slices = len(self.slices)
        side = int(math.sqrt(n_slices))
        rows, cols = side, side
        
        # Si no es cuadrado perfecto, intentamos ajustar (ej. 2x3 para 6 piezas)
        if rows * cols != n_slices:
            # Lógica simple: si no es cuadrado, asumimos que es ancho
            # Esto se puede mejorar si conoces las dimensiones
            pass 

        grid = [[None for _ in range(cols)] for _ in range(rows)]
        used_indices = set()
        
        # 1. Detectar esquina
        print("Buscando la esquina superior izquierda...")
        start_idx = self.find_top_left_corner(n_slices)
        grid[0][0] = self.slices[start_idx]
        used_indices.add(start_idx)
        print(f"-> Pieza inicial seleccionada: {self.slices[start_idx].filename}")
        
        # 2. Rellenar grid
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0: continue
                
                best_idx = -1
                min_cost = float('inf')
                
                for idx in range(n_slices):
                    if idx in used_indices: continue
                    
                    cost = 0
                    count = 0
                    
                    if c > 0: # Comparar con vecino izquierdo
                        left_slice = grid[r][c-1]
                        cost += self.calculate_cost(left_slice.id, idx, 'horizontal')
                        count += 1
                        
                    if r > 0: # Comparar con vecino superior
                        top_slice = grid[r-1][c]
                        cost += self.calculate_cost(top_slice.id, idx, 'vertical')
                        count += 1
                    
                    avg_cost = cost / count if count > 0 else float('inf')
                    
                    if avg_cost < min_cost:
                        min_cost = avg_cost
                        best_idx = idx
                
                # Fallback de seguridad
                if best_idx == -1:
                    best_idx = next(i for i in range(n_slices) if i not in used_indices)
                
                grid[r][c] = self.slices[best_idx]
                used_indices.add(best_idx)
                
        self.save_results(grid, rows, cols)

    def save_results(self, grid, rows, cols):
        # Crear carpeta específica para esta imagen y número de slices
        num_slices = len(self.slices)
        image_folder = f"{self.image_name}_{num_slices}slices" if self.image_name else f"imagen_{num_slices}slices"
        specific_output_dir = os.path.join(self.output_dir, image_folder)
        os.makedirs(specific_output_dir, exist_ok=True)
        
        h, w = self.slices[0].image.shape[:2]
        canvas = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        
        # Generar nombres de archivo más simples ya que están en carpeta específica
        method_name = self.__class__.__name__.replace('Solver', '').lower()
        
        map_filename = os.path.join(specific_output_dir, f"{method_name}_reconstruction_map.txt")
        img_filename = os.path.join(specific_output_dir, f"{method_name}_reconstructed.png")
        
        with open(map_filename, 'w') as f:
            f.write(f"Mapa de reconstrucción para: {self.image_name or 'imagen'}\n")
            f.write(f"Método utilizado: {method_name.upper()}\n")
            f.write(f"Número total de trozos: {num_slices}\n")
            f.write(f"Dimensiones: {rows}x{cols} trozos\n")
            f.write(f"Generado por: {self.__class__.__name__}\n")
            f.write("-" * 50 + "\n")
            f.write("POSICIÓN | ARCHIVO ORIGINAL\n")
            f.write("-" * 30 + "\n")
            
            for r in range(rows):
                for c in range(cols):
                    slc = grid[r][c]
                    f.write(f"({r},{c}) -> {slc.filename}\n")
                    canvas[r*h:(r+1)*h, c*w:(c+1)*w] = slc.image
        
        cv2.imwrite(img_filename, canvas)
        print(f"✓ Imagen guardada: {img_filename}")
        print(f"✓ Mapa guardado: {map_filename}")
