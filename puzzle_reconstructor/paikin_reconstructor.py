import cv2
import numpy as np
import os
import sys
import heapq
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from puzzle_base import PuzzleSolverBase, ImageSlice

class PaikinSolver(PuzzleSolverBase):
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = "", border_width: int = 10):
        super().__init__(sliced_dir, output_dir, image_name, border_width)
        self._cost_cache = {}
        self._compatibility_matrix = None
        self._edge_scores = {}
        
        # Parameters
        self.feature_border_width = border_width  # Use configurable border_width
        self.compatibility_top_k = 15
        self.multi_border_width = 3
        
        # Gradient activity threshold to consider an area as flat background
        self.bg_activity_threshold = 0.03

        # Base weights for objects (adjusted dynamically)
        self.weight_color = 0.4
        self.weight_gradient = 0.6

    # (Analysis methods and automatic configuration are maintained from original code)
    # ...

    # =============================
    #   EXTERNAL EDGE DETECTION (Relaxed)
    # =============================
    def detect_puzzle_edges(self):
        # In images with large background areas, automatic external edge detection
        # often fails and marks internal white pieces as edges. Disable it here.
        print(f"[{self.__class__.__name__}] External edge detection relaxed for safety.")
        for i, slice_obj in enumerate(self.slices):
            self._edge_scores[i] = {'top': False, 'bottom': False, 'left': False, 'right': False}

    def find_corner_piece(self, corner_type: str = 'top_left'):
        # Without using edge detection, start from piece 0.
        # The Best-First algorithm will correct placement.
        return 0

    # =============================
    #   FEATURES (Simplified)
    # =============================
    def extract_features(self, img: np.ndarray):
        """
        Use only LAB (color) and Gradient Magnitude (activity).
        Remove textures and other channels that add noise on flat backgrounds.
        """
        h, w = img.shape[:2]
        
        # 1. LAB color space normalized to [0,1]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        
        # 2. Gradient Magnitude (Scharr) normalized to [0,1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        g_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        grad_mag = cv2.magnitude(g_x, g_y)
        # Clip extreme values to avoid a bright spot dominating
        grad_mag = np.clip(grad_mag, 0, 1.0) 
        
        # Combine: 3 LAB channels + 1 magnitude channel = 4 channels
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
    #   COMPATIBILITY COST (Dynamic)
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
        
        # Flatten [N pixels, 4 channels]
        edge_a_flat = edge_a.reshape(-1, edge_a.shape[-1])
        edge_b_flat = edge_b.reshape(-1, edge_b.shape[-1])
        
        # --- DYNAMIC LOGIC ---
        
        # 1. Measure "activity" using the magnitude channel (index 3)
        act_a = np.mean(edge_a_flat[:, 3])
        act_b = np.mean(edge_b_flat[:, 3])
        
        is_bg_a = act_a < self.bg_activity_threshold
        is_bg_b = act_b < self.bg_activity_threshold
        
        # 2. Compute base differences
        # Color difference (Channels 0,1,2 - LAB)
        diff_color = np.linalg.norm(edge_a_flat[:, :3] - edge_b_flat[:, :3], axis=1).mean()
        # Gradient difference (Channel 3 - Magnitude)
        diff_grad = np.abs(edge_a_flat[:, 3] - edge_b_flat[:, 3]).mean()
        
        cost = 0.0
        
        # CASE 1: BOTH ARE BACKGROUND (background with background)
        if is_bg_a and is_bg_b:
            # Only color continuity matters. Gradient is noise.
            cost = diff_color * 2.0 # 100% weight to color
            
        # CASE 2: MIXED (background with object)
        elif is_bg_a != is_bg_b:
            # High penalty.
            cost = 5.0 + diff_color + diff_grad
            
        # CASE 3: BOTH ARE OBJECTS
        else:
            # Weighted combination of color and gradient shape.
            cost = (self.weight_color * diff_color) + (self.weight_gradient * diff_grad)

        self._cost_cache[cache_key] = cost
        return cost

    # (IMPORTANT NOTE: The remaining methods of the original PaikinSolver class:
    #  build_compatibility_matrix, get_best_candidates, solve, _evaluate_slot_fit,
    #  _add_candidates_for_slot, save_results and the if __name__ block
    #  MUST REMAIN EXACTLY THE SAME as in the original code you provided.
    #  The assembly logic does not change, only how the join cost is calculated.)