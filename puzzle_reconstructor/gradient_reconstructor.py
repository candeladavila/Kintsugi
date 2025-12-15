import cv2
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional

# Asegurar importación de puzzle_base
sys.path.insert(0, os.path.dirname(__file__))
from puzzle_base import PuzzleSolverBase, ImageSlice


class GradientSolver(PuzzleSolverBase):
    """
    Solver de gradientes mejorado con análisis de continuidad de bordes en color.
    
    Metodología:
    1. Extrae bordes del ancho especificado de cada pieza
    2. Genera imágenes combinadas de bordes para cada par compatible
    3. Analiza gradientes en espacio LAB (color) para mejor precisión
    4. Asigna índice de calidad basado en continuidad de gradientes
    5. Realiza matching óptimo usando algoritmo greedy
    """
    
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = "", border_width: int = 10):
        super().__init__(sliced_dir, output_dir, image_name, border_width)
        self.border_width = 100  # Ancho de los bordes a extraer
        self.compatibility_matrix = {}  # Matriz de compatibilidad entre piezas
        
    def extract_border(self, img: np.ndarray, side: str) -> np.ndarray:
        """
        Extrae un borde de 100 píxeles de ancho de la imagen.
        
        Args:
            img: Imagen de la que extraer el borde
            side: 'top', 'bottom', 'left', 'right'
            
        Returns:
            Array con el borde extraído
        """
        h, w = img.shape[:2]
        border_w = min(self.border_width, w, h)  # Ajustar si la imagen es muy pequeña
        
        if side == 'top':
            return img[0:border_w, :].copy()
        elif side == 'bottom':
            return img[-border_w:, :].copy()
        elif side == 'left':
            return img[:, 0:border_w].copy()
        elif side == 'right':
            return img[:, -border_w:].copy()
        else:
            raise ValueError(f"Lado inválido: {side}")
    
    def create_border_pair_image(self, border1: np.ndarray, border2: np.ndarray, 
                                  orientation: str) -> np.ndarray:
        """
        Combina dos bordes en una sola imagen para análisis de continuidad.
        
        Args:
            border1: Primer borde (pieza origen)
            border2: Segundo borde (pieza destino)
            orientation: 'horizontal' o 'vertical'
            
        Returns:
            Imagen combinada con ambos bordes unidos
        """
        if orientation == 'horizontal':
            # Unir horizontalmente (izquierda-derecha)
            # border1 (derecho de pieza A) + border2 (izquierdo de pieza B)
            combined = np.hstack([border1, border2])
        elif orientation == 'vertical':
            # Unir verticalmente (arriba-abajo)
            # border1 (inferior de pieza A) + border2 (superior de pieza B)
            combined = np.vstack([border1, border2])
        else:
            raise ValueError(f"Orientación inválida: {orientation}")
        
        return combined
    
    def calculate_gradient_continuity(self, combined_border: np.ndarray, 
                                       orientation: str) -> float:
        """
        Calcula la continuidad del gradiente en la zona de unión de dos bordes.
        Analiza gradientes en COLOR (cada canal por separado) para mejor precisión.
        
        Args:
            combined_border: Imagen con dos bordes unidos (BGR)
            orientation: 'horizontal' o 'vertical'
            
        Returns:
            Índice de calidad (menor = mejor continuidad)
        """
        # Convertir a LAB para análisis perceptual
        lab = cv2.cvtColor(combined_border, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # También mantener RGB para análisis adicional
        rgb = combined_border.astype(np.float32)
        
        # Calcular gradientes en cada canal LAB (más perceptual)
        gradients_lab = []
        for channel in range(3):
            grad_x = cv2.Sobel(lab[:, :, channel], cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(lab[:, :, channel], cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            gradients_lab.append(grad_mag)
        
        # Magnitud del gradiente combinado (promedio ponderado de canales LAB)
        # L (luminancia) tiene más peso, a y b (cromaticidad) menos
        grad_magnitude = (gradients_lab[0] * 0.6 + 
                         gradients_lab[1] * 0.2 + 
                         gradients_lab[2] * 0.2)
        
        # Analizar la continuidad en la línea de unión
        h, w = combined_border.shape[:2]
        
        if orientation == 'horizontal':
            # La unión está en el centro vertical (mitad del ancho)
            center = w // 2
            
            # Extraer región alrededor de la unión (±15 píxeles para análisis más amplio)
            region_width = 30
            left_bound = max(0, center - region_width // 2)
            right_bound = min(w, center + region_width // 2)
            
            junction_region_grad = grad_magnitude[:, left_bound:right_bound]
            junction_region_lab = lab[:, left_bound:right_bound, :]
            
            # Calcular diferencias de gradiente en la unión
            center_local = (right_bound - left_bound) // 2
            if center_local > 0 and center_local < junction_region_grad.shape[1] - 1:
                left_grad = junction_region_grad[:, center_local - 1]
                right_grad = junction_region_grad[:, center_local + 1]
                
                # Diferencia absoluta promedio en gradiente
                diff_grad = np.mean(np.abs(left_grad - right_grad))
                
                # Diferencia en color LAB en la línea de unión
                left_color = junction_region_lab[:, center_local - 1, :]
                right_color = junction_region_lab[:, center_local + 1, :]
                diff_color = np.mean(np.sqrt(np.sum((left_color - right_color)**2, axis=1)))
            else:
                diff_grad = np.mean(junction_region_grad)
                diff_color = 100.0
            
        elif orientation == 'vertical':
            # La unión está en el centro horizontal (mitad de la altura)
            center = h // 2
            
            # Extraer región alrededor de la unión
            region_height = 30
            top_bound = max(0, center - region_height // 2)
            bottom_bound = min(h, center + region_height // 2)
            
            junction_region_grad = grad_magnitude[top_bound:bottom_bound, :]
            junction_region_lab = lab[top_bound:bottom_bound, :, :]
            
            # Calcular diferencias de gradiente en la unión
            center_local = (bottom_bound - top_bound) // 2
            if center_local > 0 and center_local < junction_region_grad.shape[0] - 1:
                top_grad = junction_region_grad[center_local - 1, :]
                bottom_grad = junction_region_grad[center_local + 1, :]
                
                # Diferencia absoluta promedio en gradiente
                diff_grad = np.mean(np.abs(top_grad - bottom_grad))
                
                # Diferencia en color LAB en la línea de unión
                top_color = junction_region_lab[center_local - 1, :, :]
                bottom_color = junction_region_lab[center_local + 1, :, :]
                diff_color = np.mean(np.sqrt(np.sum((top_color - bottom_color)**2, axis=1)))
            else:
                diff_grad = np.mean(junction_region_grad)
                diff_color = 100.0
        else:
            diff_grad = float('inf')
            diff_color = float('inf')
        
        # Calcular índice de calidad adicional: suavidad general en la unión
        smoothness_grad = np.std(junction_region_grad)
        smoothness_color = np.std(junction_region_lab)
        
        # Combinar métricas con pesos ajustados:
        # - Diferencia de gradiente (40%)
        # - Diferencia de color (40%)
        # - Suavidad de gradiente (10%)
        # - Suavidad de color (10%)
        quality_index = (0.4 * diff_grad + 
                        0.4 * diff_color + 
                        0.1 * smoothness_grad + 
                        0.1 * smoothness_color)
        
        return quality_index
    
    def calculate_compatibility(self, piece_a_idx: int, piece_b_idx: int, 
                                 relation: str) -> float:
        """
        Calcula la compatibilidad entre dos piezas en una relación específica.
        
        Args:
            piece_a_idx: Índice de la pieza A
            piece_b_idx: Índice de la pieza B
            relation: 'right' (A está a la izq de B), 'bottom' (A está arriba de B)
            
        Returns:
            Índice de calidad (menor = mejor match)
        """
        img_a = self.slices[piece_a_idx].image
        img_b = self.slices[piece_b_idx].image
        
        if relation == 'right':
            # A está a la izquierda de B: comparar borde derecho de A con izquierdo de B
            border_a = self.extract_border(img_a, 'right')
            border_b = self.extract_border(img_b, 'left')
            orientation = 'horizontal'
            
        elif relation == 'bottom':
            # A está arriba de B: comparar borde inferior de A con superior de B
            border_a = self.extract_border(img_a, 'bottom')
            border_b = self.extract_border(img_b, 'top')
            orientation = 'vertical'
            
        else:
            raise ValueError(f"Relación inválida: {relation}")
        
        # Crear imagen combinada de bordes
        combined = self.create_border_pair_image(border_a, border_b, orientation)
        
        # Calcular continuidad del gradiente
        quality = self.calculate_gradient_continuity(combined, orientation)
        
        return quality
    
    def build_compatibility_matrix(self):
        """
        Construye la matriz de compatibilidad entre todas las piezas.
        Calcula el índice de calidad para cada posible combinación.
        """
        n = len(self.slices)
        print(f"[{self.__class__.__name__}] Construyendo matriz de compatibilidad...")
        print(f"  Analizando {n} piezas con bordes de {self.border_width}px...")
        
        # Inicializar matriz
        self.compatibility_matrix = {
            'right': np.full((n, n), float('inf')),  # [i, j] = calidad de i->j (horizontal)
            'bottom': np.full((n, n), float('inf'))  # [i, j] = calidad de i->j (vertical)
        }
        
        # Calcular compatibilidades
        total_comparisons = n * (n - 1) * 2
        current = 0
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # Compatibilidad horizontal (i a la izquierda de j)
                quality_h = self.calculate_compatibility(i, j, 'right')
                self.compatibility_matrix['right'][i, j] = quality_h
                
                # Compatibilidad vertical (i arriba de j)
                quality_v = self.calculate_compatibility(i, j, 'bottom')
                self.compatibility_matrix['bottom'][i, j] = quality_v
                
                current += 2
                if current % 50 == 0:
                    progress = (current / total_comparisons) * 100
                    print(f"  Progreso: {progress:.1f}% ({current}/{total_comparisons})")
        
        print(f"  ✓ Matriz de compatibilidad completada\n")
    
    def find_best_match(self, piece_idx: int, relation: str, 
                        used_pieces: set) -> Tuple[int, float]:
        """
        Encuentra la mejor pieza para emparejar con una pieza dada.
        
        Args:
            piece_idx: Índice de la pieza origen
            relation: 'right' o 'bottom'
            used_pieces: Conjunto de piezas ya utilizadas
            
        Returns:
            Tupla (índice_mejor_match, calidad)
        """
        n = len(self.slices)
        best_idx = -1
        best_quality = float('inf')
        
        for j in range(n):
            if j in used_pieces or j == piece_idx:
                continue
            
            quality = self.compatibility_matrix[relation][piece_idx, j]
            
            if quality < best_quality:
                best_quality = quality
                best_idx = j
        
        return best_idx, best_quality
    
    def solve_greedy(self, rows: int, cols: int) -> List[List[Optional[ImageSlice]]]:
        """
        Resuelve el puzzle usando un algoritmo greedy basado en mejor compatibilidad.
        
        Args:
            rows: Número de filas
            cols: Número de columnas
            
        Returns:
            Grid con las piezas organizadas
        """
        grid = [[None for _ in range(cols)] for _ in range(rows)]
        used_pieces = set()
        
        print(f"[{self.__class__.__name__}] Iniciando reconstrucción greedy...")
        print(f"  Dimensiones: {rows}x{cols}\n")
        
        # Estrategia: construir fila por fila, de izquierda a derecha
        
        # Colocar primera pieza (esquina superior izquierda)
        # Usar la pieza que tenga mejores bordes externos (menos gradiente en top y left)
        first_piece = self.find_corner_piece()
        grid[0][0] = self.slices[first_piece]
        used_pieces.add(first_piece)
        print(f"  Esquina inicial: pieza #{first_piece} ({self.slices[first_piece].filename})")
        
        # Construir el grid
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] is not None:
                    continue
                
                # Determinar de dónde viene la restricción más fuerte
                candidates = []
                
                # Si hay pieza a la izquierda, buscar mejor match a la derecha
                if c > 0 and grid[r][c-1] is not None:
                    left_piece_idx = grid[r][c-1].id
                    best_right, quality_h = self.find_best_match(left_piece_idx, 'right', used_pieces)
                    if best_right >= 0:
                        candidates.append((best_right, quality_h, 'horizontal'))
                
                # Si hay pieza arriba, buscar mejor match abajo
                if r > 0 and grid[r-1][c] is not None:
                    top_piece_idx = grid[r-1][c].id
                    best_bottom, quality_v = self.find_best_match(top_piece_idx, 'bottom', used_pieces)
                    if best_bottom >= 0:
                        candidates.append((best_bottom, quality_v, 'vertical'))
                
                # Seleccionar el candidato con mejor calidad
                if candidates:
                    candidates.sort(key=lambda x: x[1])  # Ordenar por calidad (menor = mejor)
                    best_idx = candidates[0][0]
                    
                    # Verificar si el candidato satisface ambas restricciones si existen
                    if len(candidates) > 1:
                        # Hay restricción desde arriba y desde la izquierda
                        # Validar que la pieza sea compatible con ambas
                        valid = True
                        for cand_idx, cand_quality, cand_dir in candidates:
                            if cand_idx != best_idx:
                                # Verificar que best_idx también sea razonablemente bueno en otra dirección
                                if cand_dir == 'horizontal' and c > 0:
                                    left_idx = grid[r][c-1].id
                                    alt_quality = self.compatibility_matrix['right'][left_idx, best_idx]
                                    if alt_quality > cand_quality * 2:  # Umbral de tolerancia
                                        valid = False
                                elif cand_dir == 'vertical' and r > 0:
                                    top_idx = grid[r-1][c].id
                                    alt_quality = self.compatibility_matrix['bottom'][top_idx, best_idx]
                                    if alt_quality > cand_quality * 2:
                                        valid = False
                        
                        if not valid:
                            # Si no es válido, intentar con el segundo mejor
                            if len(candidates) > 1:
                                best_idx = candidates[1][0]
                    
                    grid[r][c] = self.slices[best_idx]
                    used_pieces.add(best_idx)
                    
                    progress = len(used_pieces) / len(self.slices) * 100
                    print(f"  [{progress:5.1f}%] Posición ({r},{c}): pieza #{best_idx} (calidad: {candidates[0][1]:.2f})")
                else:
                    # No hay restricciones previas, usar pieza no utilizada
                    for idx in range(len(self.slices)):
                        if idx not in used_pieces:
                            grid[r][c] = self.slices[idx]
                            used_pieces.add(idx)
                            print(f"  Posición ({r},{c}): pieza #{idx} (sin restricciones previas)")
                            break
        
        print(f"\n  ✓ Reconstrucción completada: {len(used_pieces)}/{len(self.slices)} piezas\n")
        return grid
    
    def find_corner_piece(self) -> int:
        """
        Encuentra la pieza que probablemente sea una esquina.
        Busca la pieza con menor gradiente y menor variación de color en bordes externos.
        Usa análisis de color para mejorar la detección.
        
        Returns:
            Índice de la mejor pieza para esquina superior izquierda
        """
        best_idx = 0
        best_score = float('inf')
        
        for idx, slice_obj in enumerate(self.slices):
            img = slice_obj.image
            
            # Extraer bordes top y left
            top_border = self.extract_border(img, 'top')
            left_border = self.extract_border(img, 'left')
            
            # Convertir a LAB para análisis perceptual
            top_lab = cv2.cvtColor(top_border, cv2.COLOR_BGR2LAB).astype(np.float32)
            left_lab = cv2.cvtColor(left_border, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # Calcular gradiente promedio en estos bordes (canal L principalmente)
            grad_top_l = np.mean(np.abs(cv2.Sobel(top_lab[:,:,0], cv2.CV_64F, 1, 0)))
            grad_left_l = np.mean(np.abs(cv2.Sobel(left_lab[:,:,0], cv2.CV_64F, 0, 1)))
            
            # Calcular variación de color (menor variación = más probable borde externo)
            color_var_top = np.std(top_lab)
            color_var_left = np.std(left_lab)
            
            # Score combinado: gradiente + variación de color
            # Menor score = más probable esquina/borde externo
            score = (grad_top_l + grad_left_l) * 0.6 + (color_var_top + color_var_left) * 0.4
            
            if score < best_score:
                best_score = score
                best_idx = idx
        
        return best_idx
    
    def solve(self):
        """
        Método principal de resolución.
        """
        import math
        
        n = len(self.slices)
        side = int(math.sqrt(n))
        rows, cols = side, side
        
        print(f"\n{'='*60}")
        print(f"GRADIENT RECONSTRUCTOR - Análisis de Continuidad en Color")
        print(f"{'='*60}")
        print(f"Piezas: {n} ({rows}x{cols})")
        print(f"Ancho de bordes: {self.border_width}px")
        print(f"{'='*60}\n")
        
        # Paso 1: Construir matriz de compatibilidad
        self.build_compatibility_matrix()
        
        # Paso 2: Resolver con algoritmo greedy
        grid = self.solve_greedy(rows, cols)
        
        # Paso 3: Guardar resultados
        self.save_results(grid, rows, cols)
        
        print(f"{'='*60}")
        print(f"✓ Proceso completado")
        print(f"{'='*60}\n")


def main():
    """
    Función de prueba para ejecutar el solver de forma independiente.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Gradient Reconstructor - Análisis de continuidad de bordes en color')
    parser.add_argument('sliced_dir', help='Directorio con las piezas cortadas')
    parser.add_argument('--output', '-o', default='output_images', help='Directorio de salida')
    parser.add_argument('--name', '-n', default='imagen', help='Nombre base de la imagen')
    
    args = parser.parse_args()
    
    solver = GradientSolver(args.sliced_dir, args.output, args.name)
    solver.load_slices(args.name)
    solver.solve()


if __name__ == "__main__":
    main()
