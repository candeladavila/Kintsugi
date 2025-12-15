import cv2
import numpy as np
import os
import sys
import heapq
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from puzzle_base import PuzzleSolverBase, ImageSlice

class PaikinSolver(PuzzleSolverBase):
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = "", border_width: int = 30):
        super().__init__(sliced_dir, output_dir, image_name, border_width)
        self._cost_cache = {}
        self._compatibility_matrix = None
        self._edge_scores = {}
        
        # Parámetros
        self.feature_border_width = 4
        self.compatibility_top_k = 15
        self.multi_border_width = 3
        
        # Umbral de actividad del gradiente para considerar que es fondo plano
        self.bg_activity_threshold = 0.03

        # Pesos base para objetos (se ajustan dinámicamente)
        self.weight_color = 0.4
        self.weight_gradient = 0.6

    # (Analysis methods and automatic configuration are maintained from original code)
    # ...

    # =============================
    #   DETECCIÓN DE BORDES EXTERNOS (Relajada)
    # =============================
    def detect_puzzle_edges(self):
        # En imágenes con mucho fondo, la detección automática de bordes externos 
        # suele fallar y marcar piezas internas blancas como bordes. La desactivamos.
        print(f"[{self.__class__.__name__}] Detección de bordes externos relajada por seguridad.")
        for i, slice_obj in enumerate(self.slices):
             self._edge_scores[i] = {'top': False, 'bottom': False, 'left': False, 'right': False}

    def find_corner_piece(self, corner_type: str = 'top_left'):
        # Al no usar detección de bordes, empezamos por la pieza 0. 
        # El algoritmo Best-First corregirá la posición.
        return 0

    # =============================
    #   FEATURES (Simplificadas)
    # =============================
    def extract_features(self, img: np.ndarray):
        """
        Solo usamos LAB (color) y Magnitud de Gradiente (actividad).
        Eliminamos texturas y otros espacios que añaden ruido en fondos planos.
        """
        h, w = img.shape[:2]
        
        # 1. Espacio LAB normalizado [0,1]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        
        # 2. Magnitud de Gradiente (Scharr) normalizada [0,1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        g_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        grad_mag = cv2.magnitude(g_x, g_y)
        # Recortamos valores extremos para evitar que un punto brillante domine
        grad_mag = np.clip(grad_mag, 0, 1.0) 
        
        # Combinar: 3 canales LAB + 1 canal Magnitud = 4 canales
        combined = np.dstack([
            lab,                # Canales 0, 1, 2
            grad_mag[..., None] # Canal 3
        ]).astype(np.float32)

        w_b = self.feature_border_width
        w_b = min(w_b, h // 2, w // 2)
        return {
            'top': combined[0:w_b, :, :],
            'bottom': combined[-w_b:, :, :],
            'left': combined[:, 0:w_b, :],
            'right': combined[:, -w_b:, :]
        }

    # =============================
    #   COSTE DE COMPATIBILIDAD (Dinámico)
    # =============================
    def calculate_cost(self, idx_a: int, idx_b: int, direction: str) -> float:
        cache_key = (idx_a, idx_b, direction)
        if cache_key in self._cost_cache: return self._cost_cache[cache_key]

        feats_a = self.slices[idx_a].borders
        feats_b = self.slices[idx_b].borders
        mbw = self.multi_border_width
        
        if direction == 'horizontal':
            max_w = min(feats_a['right'].shape[1], feats_b['left'].shape[1], mbw)
            edge_a = feats_a['right'][:, -max_w:, :]
            edge_b = feats_b['left'][:, :max_w, :]
        else: # vertical
            max_h = min(feats_a['bottom'].shape[0], feats_b['top'].shape[0], mbw)
            edge_a = feats_a['bottom'][-max_h:, :, :]
            edge_b = feats_b['top'][:max_h, :, :]
        
        # Flatten [N píxeles, 4 canales]
        edge_a_flat = edge_a.reshape(-1, edge_a.shape[-1])
        edge_b_flat = edge_b.reshape(-1, edge_b.shape[-1])
        
        # --- DYNAMIC LOGIC ---
        
        # 1. Measure "Activity" using magnitude channel (index 3)
        act_a = np.mean(edge_a_flat[:, 3])
        act_b = np.mean(edge_b_flat[:, 3])
        
        is_bg_a = act_a < self.bg_activity_threshold
        is_bg_b = act_b < self.bg_activity_threshold
        
        # 2. Calcular diferencias base
        # Diferencia de Color (Canales 0,1,2 - LAB)
        diff_color = np.linalg.norm(edge_a_flat[:, :3] - edge_b_flat[:, :3], axis=1).mean()
        # Diferencia de Gradiente (Canal 3 - Magnitud)
        diff_grad = np.abs(edge_a_flat[:, 3] - edge_b_flat[:, 3]).mean()
        
        cost = 0.0
        
        # CASO 1: AMBOS SON FONDO (Blanco con Blanco)
        if is_bg_a and is_bg_b:
            # Solo importa la continuidad del color. El gradiente es ruido.
            cost = diff_color * 2.0 # Peso 100% al color
            
        # CASO 2: MIXTO (Fondo con Objeto)
        elif is_bg_a != is_bg_b:
            # Penalización alta.
            cost = 5.0 + diff_color + diff_grad
            
        # CASO 3: AMBOS SON OBJETOS (Bordes de manzana)
        else:
            # Combinación ponderada de color y forma del gradiente.
            cost = (self.weight_color * diff_color) + (self.weight_gradient * diff_grad)

        self._cost_cache[cache_key] = cost
        return cost

    # (NOTA IMPORTANTE: Los métodos restantes de la clase PaikinSolver original:
    #  build_compatibility_matrix, get_best_candidates, solve, _evaluate_slot_fit, 
    #  _add_candidates_for_slot, save_results y el bloque if __name__ 
    #  DEBEN MANTENERSE EXACTAMENTE IGUAL que en el código que proporcionaste al principio.
    #  La lógica de ensamblado no cambia, solo cómo se calcula el coste de unión).