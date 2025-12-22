import cv2
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional

# Ensure puzzle_base import
sys.path.insert(0, os.path.dirname(__file__))
from puzzle_base import PuzzleSolverBase, ImageSlice


class GradientSolver(PuzzleSolverBase):
    """
    Enhanced gradient solver with edge continuity analysis in color.

    Methodology:
    1. Extract borders of the specified width from each piece
    2. Generate combined border images for each candidate pair
    3. Analyze gradients in LAB (color) space for improved accuracy
    4. Assign a quality index based on gradient continuity
    5. Perform matching using a greedy algorithm
    """
    
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = "", border_width: int = 100):
        super().__init__(sliced_dir, output_dir, image_name, border_width)
        # border_width is now configurable via parameter (default 100px)
        self.compatibility_matrix = {}  # Compatibility matrix between pieces
        
    def extract_border(self, img: np.ndarray, side: str) -> np.ndarray:
        """
        Extract a border of the configured width from the image.

        Args:
            img: Image to extract the border from
            side: 'top', 'bottom', 'left', 'right'

        Returns:
            Array with the extracted border
        """
        h, w = img.shape[:2]
        border_w = min(self.border_width, w, h)  # Adjust if the image is very small
        
        if side == 'top':
            return img[0:border_w, :].copy()
        elif side == 'bottom':
            return img[-border_w:, :].copy()
        elif side == 'left':
            return img[:, 0:border_w].copy()
        elif side == 'right':
            return img[:, -border_w:].copy()
        else:
            raise ValueError(f"Invalid side: {side}")
    
    def create_border_pair_image(self, border1: np.ndarray, border2: np.ndarray, orientation: str) -> np.ndarray:
        """
        Combine two borders into a single image for continuity analysis.

        Args:
            border1: First border (source piece)
            border2: Second border (destination piece)
            orientation: 'horizontal' or 'vertical'

        Returns:
            Combined image with both borders joined
        """
        if orientation == 'horizontal':
            # Join horizontally (left-right)
            # border1 (right of piece A) + border2 (left of piece B)
            combined = np.hstack([border1, border2])
        elif orientation == 'vertical':
            # Join vertically (top-bottom)
            # border1 (bottom of piece A) + border2 (top of piece B)
            combined = np.vstack([border1, border2])
        else:
            raise ValueError(f"Invalid orientation: {orientation}")
        
        return combined
    
    def calculate_gradient_continuity(self, combined_border: np.ndarray, orientation: str) -> float:
        """
        Calculate gradient continuity in the junction area of two borders.
        Analyzes gradients in COLOR (each channel separately) for better accuracy.

        Args:
            combined_border: Image with two borders joined (BGR)
            orientation: 'horizontal' or 'vertical'

        Returns:
            Quality index (lower = better continuity)
        """
        # Convert to LAB for perceptual analysis
        lab = cv2.cvtColor(combined_border, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Also keep RGB for additional analysis
        rgb = combined_border.astype(np.float32)

        # Compute gradients in each LAB channel (more perceptual)
        gradients_lab = []
        for channel in range(3):
            grad_x = cv2.Sobel(lab[:, :, channel], cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(lab[:, :, channel], cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            gradients_lab.append(grad_mag)
        
        # Combined gradient magnitude (weighted average of LAB channels)
        # L (luminance) has more weight, a and b (chromaticity) less
        grad_magnitude = (gradients_lab[0] * 0.6 + 
                         gradients_lab[1] * 0.2 + 
                         gradients_lab[2] * 0.2)
        
        # Analyze continuity at the joining line
        h, w = combined_border.shape[:2]
        
        if orientation == 'horizontal':
            # The junction is at the vertical center (half of the width)
            center = w // 2
            
            # Extract region around the junction (±15 pixels for wider analysis)
            region_width = 30
            left_bound = max(0, center - region_width // 2)
            right_bound = min(w, center + region_width // 2)
            
            junction_region_grad = grad_magnitude[:, left_bound:right_bound]
            junction_region_lab = lab[:, left_bound:right_bound, :]
            
            # Compute gradient differences at the junction
            center_local = (right_bound - left_bound) // 2
            if center_local > 0 and center_local < junction_region_grad.shape[1] - 1:
                left_grad = junction_region_grad[:, center_local - 1]
                right_grad = junction_region_grad[:, center_local + 1]
                
                # Average absolute difference in gradient
                diff_grad = np.mean(np.abs(left_grad - right_grad))

                # Color LAB difference at the junction line
                left_color = junction_region_lab[:, center_local - 1, :]
                right_color = junction_region_lab[:, center_local + 1, :]
                diff_color = np.mean(np.sqrt(np.sum((left_color - right_color)**2, axis=1)))
            else:
                diff_grad = np.mean(junction_region_grad)
                diff_color = 100.0
            
        elif orientation == 'vertical':
            # The junction is at the horizontal center (half of the height)
            center = h // 2
            
            # Extract region around the junction
            region_height = 30
            top_bound = max(0, center - region_height // 2)
            bottom_bound = min(h, center + region_height // 2)
            
            junction_region_grad = grad_magnitude[top_bound:bottom_bound, :]
            junction_region_lab = lab[top_bound:bottom_bound, :, :]
            
            # Compute gradient differences at the junction
            center_local = (bottom_bound - top_bound) // 2
            if center_local > 0 and center_local < junction_region_grad.shape[0] - 1:
                top_grad = junction_region_grad[center_local - 1, :]
                bottom_grad = junction_region_grad[center_local + 1, :]
                
                # Average absolute difference in gradient
                diff_grad = np.mean(np.abs(top_grad - bottom_grad))

                # Color LAB difference at the junction line
                top_color = junction_region_lab[center_local - 1, :, :]
                bottom_color = junction_region_lab[center_local + 1, :, :]
                diff_color = np.mean(np.sqrt(np.sum((top_color - bottom_color)**2, axis=1)))
            else:
                diff_grad = np.mean(junction_region_grad)
                diff_color = 100.0
        else:
            diff_grad = float('inf')
            diff_color = float('inf')
        
        # Compute additional quality index: overall smoothness at the junction
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
    
    def calculate_compatibility(self, piece_a_idx: int, piece_b_idx: int, relation: str) -> float:
        """
        Calculate compatibility between two pieces for a specific relation.

        Args:
            piece_a_idx: Index of piece A
            piece_b_idx: Index of piece B
            relation: 'right' (A is left of B), 'bottom' (A is above B)

        Returns:
            Quality index (lower = better match)
        """
        img_a = self.slices[piece_a_idx].original_image
        img_b = self.slices[piece_b_idx].original_image
        
        if relation == 'right':
            # A is left of B: compare right border of A with left border of B
            border_a = self.extract_border(img_a, 'right')
            border_b = self.extract_border(img_b, 'left')
            orientation = 'horizontal'
            
        elif relation == 'bottom':
            # A is above B: compare bottom border of A with top border of B
            border_a = self.extract_border(img_a, 'bottom')
            border_b = self.extract_border(img_b, 'top')
            orientation = 'vertical'
            
        else:
            raise ValueError(f"Invalid relation: {relation}")
        
        # Crear imagen combinada de bordes
        combined = self.create_border_pair_image(border_a, border_b, orientation)
        
        # Calcular continuidad del gradiente
        quality = self.calculate_gradient_continuity(combined, orientation)
        
        return quality
    
    def build_compatibility_matrix(self):
        """
        Build the compatibility matrix between all pieces.
        Compute the quality index for every possible combination.
        """
        n = len(self.slices)
        print(f"[{self.__class__.__name__}] Building compatibility matrix...")
        print(f"  Analyzing {n} pieces with borders of {self.border_width}px...")
        
        # Initialize matrix
        self.compatibility_matrix = {
            'right': np.full((n, n), float('inf')),  # [i, j] = quality of i->j (horizontal)
            'bottom': np.full((n, n), float('inf'))  # [i, j] = quality of i->j (vertical)
        }
        
        # Compute compatibilities
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
                    print(f"  Progress: {progress:.1f}% ({current}/{total_comparisons})")
        
        print(f"  ✓ Compatibility matrix completed\n")
    
    def find_best_match(self, piece_idx: int, relation: str, 
                        used_pieces: set) -> Tuple[int, float]:
        """
        Find the best piece to match with a given piece.

        Args:
            piece_idx: Index of the source piece
            relation: 'right' or 'bottom'
            used_pieces: Set of pieces already used

        Returns:
            Tuple (best_match_index, quality)
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
        Solve the puzzle using a greedy algorithm based on best compatibility.

        Args:
            rows: Number of rows
            cols: Number of columns

        Returns:
            Grid with pieces arranged
        """
        grid = [[None for _ in range(cols)] for _ in range(rows)]
        used_pieces = set()
        
        print(f"[{self.__class__.__name__}] Starting greedy reconstruction...")
        print(f"  Dimensions: {rows}x{cols}\n")
        
        # Strategy: build row by row, left to right

        # Place first piece (top-left corner)
        # Use the piece that has the flattest external borders (less gradient on top and left)
        first_piece = self.find_corner_piece()
        grid[0][0] = self.slices[first_piece]
        used_pieces.add(first_piece)
        print(f"  Start corner: piece #{first_piece} ({self.slices[first_piece].filename})")
        
        # Construir el grid
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] is not None:
                    continue
                
                # Determine which neighbor provides the strongest constraint
                candidates = []
                
                # If there is a piece to the left, find best match to the right
                if c > 0 and grid[r][c-1] is not None:
                    left_piece_idx = grid[r][c-1].id
                    best_right, quality_h = self.find_best_match(left_piece_idx, 'right', used_pieces)
                    if best_right >= 0:
                        candidates.append((best_right, quality_h, 'horizontal'))
                
                # If there is a piece above, find best match to the bottom
                if r > 0 and grid[r-1][c] is not None:
                    top_piece_idx = grid[r-1][c].id
                    best_bottom, quality_v = self.find_best_match(top_piece_idx, 'bottom', used_pieces)
                    if best_bottom >= 0:
                        candidates.append((best_bottom, quality_v, 'vertical'))
                
                # Select the candidate with the best quality
                if candidates:
                    candidates.sort(key=lambda x: x[1])  # Ordenar por calidad (menor = mejor)
                    best_idx = candidates[0][0]
                    
                    # Verify that the candidate satisfies both constraints if they exist
                    if len(candidates) > 1:
                        # There is constraint from top and from left
                        # Validate that the piece is compatible with both
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
                        # If not valid, try the second best
                            if len(candidates) > 1:
                                best_idx = candidates[1][0]
                    
                    grid[r][c] = self.slices[best_idx]
                    used_pieces.add(best_idx)
                    
                    progress = len(used_pieces) / len(self.slices) * 100
                    print(f"  [{progress:5.1f}%] Position ({r},{c}): piece #{best_idx} (quality: {candidates[0][1]:.2f})")
                else:
                    # No prior constraints, use an unused piece
                    for idx in range(len(self.slices)):
                        if idx not in used_pieces:
                            grid[r][c] = self.slices[idx]
                            used_pieces.add(idx)
                            print(f"  Position ({r},{c}): piece #{idx} (no prior constraints)")
                            break
        
        print(f"\n  ✓ Reconstruction completed: {len(used_pieces)}/{len(self.slices)} pieces\n")
        return grid
    
    def find_corner_piece(self) -> int:
        """
        Find the piece that is likely a corner.
        Search for the piece with the lowest gradient and color variation on external borders.
        Use color analysis to improve detection.

        Returns:
            Index of the best piece for top-left corner
        """
        best_idx = 0
        best_score = float('inf')
        
        for idx, slice_obj in enumerate(self.slices):
            img = slice_obj.original_image
            
            # Extract top and left borders
            top_border = self.extract_border(img, 'top')
            left_border = self.extract_border(img, 'left')
            
            # Convert to LAB for perceptual analysis
            top_lab = cv2.cvtColor(top_border, cv2.COLOR_BGR2LAB).astype(np.float32)
            left_lab = cv2.cvtColor(left_border, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # Compute mean gradient on these borders (L channel primarily)
            grad_top_l = np.mean(np.abs(cv2.Sobel(top_lab[:,:,0], cv2.CV_64F, 1, 0)))
            grad_left_l = np.mean(np.abs(cv2.Sobel(left_lab[:,:,0], cv2.CV_64F, 0, 1)))
            
            # Compute color variation (lower variation = more likely external border)
            color_var_top = np.std(top_lab)
            color_var_left = np.std(left_lab)
            
            # Combined score: gradient + color variation
            # Lower score = more likely corner/external border
            score = (grad_top_l + grad_left_l) * 0.6 + (color_var_top + color_var_left) * 0.4
            
            if score < best_score:
                best_score = score
                best_idx = idx
        
        return best_idx
    
    def solve(self):
        """
        Main solving method.
        """
        import math
        
        n = len(self.slices)
        side = int(math.sqrt(n))
        rows, cols = side, side
        
        print(f"\n{'='*60}")
        print(f"GRADIENT RECONSTRUCTOR - Color Continuity Analysis")
        print(f"{'='*60}")
        print(f"Pieces: {n} ({rows}x{cols})")
        print(f"Border width: {self.border_width}px")
        print(f"{'='*60}\n")
        
        # Step 1: Build compatibility matrix
        self.build_compatibility_matrix()
        
        # Step 2: Solve with greedy algorithm
        grid = self.solve_greedy(rows, cols)
        
        # Step 3: Save results
        self.save_results(grid, rows, cols)
        
        print(f"{'='*60}")
        print(f"✓ Process completed")
        print(f"{'='*60}\n")


def main():
    """
    Test function to run the solver independently.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Gradient Reconstructor - Color continuity analysis')
    parser.add_argument('sliced_dir', help='Directory with sliced pieces')
    parser.add_argument('--output', '-o', default='output_images', help='Output directory')
    parser.add_argument('--name', '-n', default='image', help='Base name of the image')
    
    args = parser.parse_args()
    
    solver = GradientSolver(args.sliced_dir, args.output, args.name)
    solver.load_slices(args.name)
    solver.solve()


if __name__ == "__main__":
    main()
