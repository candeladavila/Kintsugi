"""
Puzzle Base V2 - With Rotation Support

Base class for all puzzle reconstruction algorithms that support
piece rotation detection and correction.
"""

import os
import glob
import math
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# Rotation constants
ROTATIONS = [0, 90, 180, 270]  # Possible rotation angles


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotates an image by the specified angle.
    
    Args:
        img: Image to rotate
        angle: Rotation angle (0, 90, 180, 270)
        
    Returns:
        Rotated image
    """
    angle = angle % 360
    if angle == 0:
        return img.copy()
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"Invalid angle: {angle}")


@dataclass
class ImageSliceV2:
    """Represents a puzzle piece with rotation support."""
    id: int
    filename: str
    original_image: np.ndarray  # Original image as loaded
    current_rotation: int = 0   # Current rotation applied (0, 90, 180, 270)
    borders: Dict[str, np.ndarray] = field(default_factory=dict)
    
    @property
    def image(self) -> np.ndarray:
        """Returns the image with current rotation applied."""
        return rotate_image(self.original_image, self.current_rotation)
    
    def get_rotated_image(self, rotation: int) -> np.ndarray:
        """Returns the image with a specific rotation."""
        return rotate_image(self.original_image, rotation)
    
    def set_rotation(self, angle: int):
        """Sets the current rotation angle."""
        self.current_rotation = angle % 360


class PuzzleSolverBaseV2:
    """
    Base class for puzzle solvers with rotation support.
    
    This version considers that each piece may be rotated by 0°, 90°, 180°, or 270°
    and the solver must determine both the correct position AND rotation.
    """
    
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = "", border_width: int = 10):
        self.sliced_dir = sliced_dir
        self.output_dir = output_dir
        self.image_name = image_name
        self.slices: List[ImageSliceV2] = []
        self.border_width = border_width  # Border width to analyze (configurable, default 10 pixels)
        self.version = "V2"     # Version identifier

    def extract_features(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extracts borders for analysis.
        
        Args:
            img: Image to extract features from
            
        Returns:
            Dictionary with 'top', 'bottom', 'left', 'right' borders
        """
        w_b = self.border_width
        return {
            'top': img[0:w_b, :],
            'bottom': img[-w_b:, :],
            'left': img[:, 0:w_b],
            'right': img[:, -w_b:]
        }

    def extract_features_rotated(self, img: np.ndarray, rotation: int) -> Dict[str, np.ndarray]:
        """
        Extracts borders from a rotated version of the image.
        
        Args:
            img: Original image
            rotation: Rotation angle to apply first
            
        Returns:
            Dictionary with borders from the rotated image
        """
        rotated = rotate_image(img, rotation)
        return self.extract_features(rotated)

    def load_slices(self, original_name_pattern: str):
        """Loads existing pieces from the specified folder."""
        # Search for files with pattern: name_slice_XXX.png
        search_pattern = os.path.join(self.sliced_dir, f"{original_name_pattern}_slice_*.png")
        
        # Sort numerically to process in order
        try:
            files = sorted(glob.glob(search_pattern), 
                          key=lambda x: int(x.split('_slice_')[1].split('.')[0]))
        except IndexError:
            files = sorted(glob.glob(search_pattern))
        
        if not files:
            raise FileNotFoundError(f"No images found in: {search_pattern}")
            
        print(f"[{self.__class__.__name__}] Loading {len(files)} pieces from {self.sliced_dir}...")
        print(f"[{self.__class__.__name__}] ROTATION SUPPORT: ENABLED")
        
        for idx, fpath in enumerate(files):
            img = cv2.imread(fpath)
            if img is None: 
                continue
            
            # Store original image without rotation
            slice_obj = ImageSliceV2(
                id=idx, 
                filename=os.path.basename(fpath), 
                original_image=img,
                current_rotation=0
            )
            
            # Extract initial features (will be re-extracted when rotation is determined)
            slice_obj.borders = self.extract_features(img)
            self.slices.append(slice_obj)

    def calculate_cost(self, idx_a: int, idx_b: int, direction: str, 
                       rotation_a: int = 0, rotation_b: int = 0) -> float:
        """
        Calculates the cost of placing piece B next to piece A.
        Must be implemented by subclasses.
        
        Args:
            idx_a: Index of piece A
            idx_b: Index of piece B
            direction: 'horizontal' (A left of B) or 'vertical' (A above B)
            rotation_a: Rotation applied to piece A
            rotation_b: Rotation applied to piece B
            
        Returns:
            Cost value (lower = better match)
        """
        raise NotImplementedError("Subclasses must implement calculate_cost")

    def calculate_cost_all_rotations(self, idx_a: int, idx_b: int, direction: str,
                                      fixed_rotation_a: Optional[int] = None) -> Tuple[float, int, int]:
        """
        Finds the best rotation combination for two pieces.
        
        Args:
            idx_a: Index of piece A
            idx_b: Index of piece B
            direction: 'horizontal' or 'vertical'
            fixed_rotation_a: If provided, only test this rotation for piece A
            
        Returns:
            Tuple (best_cost, best_rotation_a, best_rotation_b)
        """
        best_cost = float('inf')
        best_rot_a = 0
        best_rot_b = 0
        
        rotations_a = [fixed_rotation_a] if fixed_rotation_a is not None else ROTATIONS
        
        for rot_a in rotations_a:
            for rot_b in ROTATIONS:
                cost = self.calculate_cost(idx_a, idx_b, direction, rot_a, rot_b)
                if cost < best_cost:
                    best_cost = cost
                    best_rot_a = rot_a
                    best_rot_b = rot_b
        
        return best_cost, best_rot_a, best_rot_b

    def find_top_left_corner(self, n_slices: int) -> Tuple[int, int]:
        """
        Finds the piece and rotation that is probably the top-left corner.
        
        Returns:
            Tuple (piece_index, rotation)
        """
        max_min_cost = -1
        best_candidate = 0
        best_rotation = 0

        for i in range(n_slices):
            for rotation in ROTATIONS:
                # Calculate minimum costs from left and top with this rotation
                min_left = min([
                    self.calculate_cost(j, i, 'horizontal', 0, rotation) 
                    for j in range(n_slices) if i != j
                ])
                min_top = min([
                    self.calculate_cost(j, i, 'vertical', 0, rotation) 
                    for j in range(n_slices) if i != j
                ])
                
                corner_score = min_left + min_top
                
                if corner_score > max_min_cost:
                    max_min_cost = corner_score
                    best_candidate = i
                    best_rotation = rotation
        
        return best_candidate, best_rotation

    def solve(self):
        """
        Automatic Greedy algorithm to reconstruct the puzzle with rotation support.
        """
        n_slices = len(self.slices)
        side = int(math.sqrt(n_slices))
        rows, cols = side, side
        
        if rows * cols != n_slices:
            print(f"Warning: {n_slices} pieces don't form a perfect square")

        # Grid stores tuples: (slice_object, rotation)
        grid = [[None for _ in range(cols)] for _ in range(rows)]
        used_indices = set()
        
        # 1. Find corner piece with best rotation
        print("Searching for top-left corner (with rotation)...")
        start_idx, start_rotation = self.find_top_left_corner(n_slices)
        self.slices[start_idx].set_rotation(start_rotation)
        grid[0][0] = self.slices[start_idx]
        used_indices.add(start_idx)
        print(f"-> Initial piece: {self.slices[start_idx].filename} (rotation: {start_rotation}°)")
        
        # 2. Fill grid
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0: 
                    continue
                
                best_idx = -1
                best_rotation = 0
                min_cost = float('inf')
                
                for idx in range(n_slices):
                    if idx in used_indices: 
                        continue
                    
                    # Try all rotations for this piece
                    for rotation in ROTATIONS:
                        cost = 0
                        count = 0
                        
                        # Compare with left neighbor
                        if c > 0:
                            left_slice = grid[r][c-1]
                            cost += self.calculate_cost(
                                left_slice.id, idx, 'horizontal',
                                left_slice.current_rotation, rotation
                            )
                            count += 1
                        
                        # Compare with top neighbor
                        if r > 0:
                            top_slice = grid[r-1][c]
                            cost += self.calculate_cost(
                                top_slice.id, idx, 'vertical',
                                top_slice.current_rotation, rotation
                            )
                            count += 1
                        
                        avg_cost = cost / count if count > 0 else float('inf')
                        
                        if avg_cost < min_cost:
                            min_cost = avg_cost
                            best_idx = idx
                            best_rotation = rotation
                
                # Safety fallback
                if best_idx == -1:
                    best_idx = next(i for i in range(n_slices) if i not in used_indices)
                    best_rotation = 0
                
                self.slices[best_idx].set_rotation(best_rotation)
                grid[r][c] = self.slices[best_idx]
                used_indices.add(best_idx)
        
        self.save_results(grid, rows, cols)

    def load_solution_mapping(self) -> dict:
        """
        Loads the correct solution mapping from the _order.txt file.
        For V2, also loads correct rotation information.
        
        Returns:
            Dictionary {filename: (correct_row, correct_col, correct_rotation)}
        """
        order_file = os.path.join(self.sliced_dir, f"{self.image_name}_order.txt")
        solution_map = {}
        
        if not os.path.exists(order_file):
            print(f"⚠ Solution file not found: {order_file}")
            return solution_map
        
        try:
            with open(order_file, 'r') as f:
                lines = f.readlines()
            
            # Check if this is a V2 file (has rotation info)
            is_v2 = any("VERSION 2" in line or "Correct Rot" in line for line in lines)
            
            in_section = False
            for line in lines:
                if "CORRECT ORDER FOR RECONSTRUCTION" in line or "ORDEN CORRECTO" in line:
                    in_section = True
                    continue
                
                if in_section and "|" in line and "_slice_" in line:
                    parts = line.split("|")
                    if len(parts) >= 4:
                        filename = parts[1].strip()
                        row = int(parts[2].strip())
                        col = int(parts[3].strip())
                        
                        # Try to extract rotation for V2
                        correct_rotation = 0
                        if is_v2 and len(parts) >= 7:
                            try:
                                rot_str = parts[6].strip().replace('°', '')
                                correct_rotation = int(rot_str)
                            except (ValueError, IndexError):
                                correct_rotation = 0
                        
                        solution_map[filename] = (row, col, correct_rotation)
                        
        except Exception as e:
            print(f"⚠ Error reading solution file: {e}")
        
        return solution_map
    
    def calculate_accuracy(self, grid, rows: int, cols: int) -> dict:
        """
        Calculates reconstruction accuracy based on correctly connected borders
        AND correct rotations.
        
        New: Also tracks "relative correct borders" - pieces that are correctly
        connected relative to each other, even if rotated together.
        
        Returns:
            Dictionary with accuracy metrics
        """
        solution_map = self.load_solution_mapping()
        
        if not solution_map:
            return {
                'correct_borders': 0,
                'total_borders': 0,
                'border_accuracy_percent': 0.0,
                'correct_rotations': 0,
                'total_pieces': 0,
                'rotation_accuracy_percent': 0.0,
                'relative_correct_borders': 0,
                'relative_border_accuracy_percent': 0.0,
                'correct_border_positions': []
            }
        
        # Inverse mapping: (correct_row, correct_col) -> filename
        position_to_file = {}
        for filename, (row, col, rotation) in solution_map.items():
            position_to_file[(row, col)] = filename
        
        correct_borders = 0
        relative_correct_borders = 0
        total_borders = 0
        correct_rotations = 0
        total_pieces = 0
        correct_border_positions = []  # Store positions of correct borders for drawing
        
        # Check correct borders and rotations
        for r in range(rows):
            for c in range(cols):
                piece = grid[r][c]
                if piece.filename not in solution_map:
                    continue
                
                total_pieces += 1
                piece_correct_row, piece_correct_col, piece_correct_rot = solution_map[piece.filename]
                
                # Check rotation accuracy
                if piece.current_rotation == piece_correct_rot:
                    correct_rotations += 1
                
                # Check right neighbor
                if c < cols - 1:
                    total_borders += 1
                    right_piece = grid[r][c + 1]
                    
                    # Check if these two pieces should be neighbors in the solution (in any direction)
                    if piece.filename in solution_map and right_piece.filename in solution_map:
                        piece_correct_row, piece_correct_col, piece_correct_rot = solution_map[piece.filename]
                        right_correct_row, right_correct_col, right_correct_rot = solution_map[right_piece.filename]
                        
                        # Calculate the vector between the two pieces in the solution
                        solution_row_diff = right_correct_row - piece_correct_row
                        solution_col_diff = right_correct_col - piece_correct_col
                        
                        # Check if they are neighbors in ANY direction in the solution (Manhattan distance = 1)
                        are_neighbors = (abs(solution_row_diff) + abs(solution_col_diff)) == 1
                        
                        if are_neighbors:
                            # Calculate rotation offsets for both pieces
                            piece_rot_offset = (piece.current_rotation - piece_correct_rot) % 360
                            right_rot_offset = (right_piece.current_rotation - right_correct_rot) % 360
                            
                            # Relative correct: both pieces have the same rotation offset
                            # This means they're correctly connected even if rotated together
                            if piece_rot_offset == right_rot_offset:
                                # Verify the spatial relationship is also correct after rotation
                                # Current vector: (0, 1) means right neighbor
                                # After rotation, check if the solution vector matches
                                
                                # Transform solution vector by the rotation offset
                                if piece_rot_offset == 0:
                                    expected_row_diff, expected_col_diff = solution_row_diff, solution_col_diff
                                elif piece_rot_offset == 90:
                                    # 90° clockwise: (row, col) -> (col, -row)
                                    expected_row_diff, expected_col_diff = solution_col_diff, -solution_row_diff
                                elif piece_rot_offset == 180:
                                    # 180°: (row, col) -> (-row, -col)
                                    expected_row_diff, expected_col_diff = -solution_row_diff, -solution_col_diff
                                elif piece_rot_offset == 270:
                                    # 270° clockwise: (row, col) -> (-col, row)
                                    expected_row_diff, expected_col_diff = -solution_col_diff, solution_row_diff
                                
                                # Current spatial relationship: right neighbor means (0, 1)
                                if expected_row_diff == 0 and expected_col_diff == 1:
                                    relative_correct_borders += 1
                                    correct_border_positions.append(('vertical', r, c))
                            
                            # Absolute correct: both have correct positions, rotations, and are right neighbors
                            if (piece.current_rotation == piece_correct_rot and 
                                right_piece.current_rotation == right_correct_rot and
                                solution_row_diff == 0 and solution_col_diff == 1):
                                correct_borders += 1
                
                # Check bottom neighbor
                if r < rows - 1:
                    total_borders += 1
                    bottom_piece = grid[r + 1][c]
                    
                    # Check if these two pieces should be neighbors in the solution (in any direction)
                    if piece.filename in solution_map and bottom_piece.filename in solution_map:
                        piece_correct_row, piece_correct_col, piece_correct_rot = solution_map[piece.filename]
                        bottom_correct_row, bottom_correct_col, bottom_correct_rot = solution_map[bottom_piece.filename]
                        
                        # Calculate the vector between the two pieces in the solution
                        solution_row_diff = bottom_correct_row - piece_correct_row
                        solution_col_diff = bottom_correct_col - piece_correct_col
                        
                        # Check if they are neighbors in ANY direction in the solution
                        are_neighbors = (abs(solution_row_diff) + abs(solution_col_diff)) == 1
                        
                        if are_neighbors:
                            # Calculate rotation offsets for both pieces
                            piece_rot_offset = (piece.current_rotation - piece_correct_rot) % 360
                            bottom_rot_offset = (bottom_piece.current_rotation - bottom_correct_rot) % 360
                            
                            # Relative correct: both pieces have the same rotation offset
                            if piece_rot_offset == bottom_rot_offset:
                                # Transform solution vector by the rotation offset
                                if piece_rot_offset == 0:
                                    expected_row_diff, expected_col_diff = solution_row_diff, solution_col_diff
                                elif piece_rot_offset == 90:
                                    expected_row_diff, expected_col_diff = solution_col_diff, -solution_row_diff
                                elif piece_rot_offset == 180:
                                    expected_row_diff, expected_col_diff = -solution_row_diff, -solution_col_diff
                                elif piece_rot_offset == 270:
                                    expected_row_diff, expected_col_diff = -solution_col_diff, solution_row_diff
                                
                                # Current spatial relationship: bottom neighbor means (1, 0)
                                if expected_row_diff == 1 and expected_col_diff == 0:
                                    relative_correct_borders += 1
                                    correct_border_positions.append(('horizontal', r, c))
                            
                            # Absolute correct: both have correct positions, rotations, and are bottom neighbors
                            if (piece.current_rotation == piece_correct_rot and 
                                bottom_piece.current_rotation == bottom_correct_rot and
                                solution_row_diff == 1 and solution_col_diff == 0):
                                correct_borders += 1
        
        border_accuracy = (correct_borders / total_borders * 100) if total_borders > 0 else 0.0
        rotation_accuracy = (correct_rotations / total_pieces * 100) if total_pieces > 0 else 0.0
        relative_border_accuracy = (relative_correct_borders / total_borders * 100) if total_borders > 0 else 0.0
        
        return {
            'correct_borders': correct_borders,
            'total_borders': total_borders,
            'border_accuracy_percent': border_accuracy,
            'correct_rotations': correct_rotations,
            'total_pieces': total_pieces,
            'rotation_accuracy_percent': rotation_accuracy,
            'relative_correct_borders': relative_correct_borders,
            'relative_border_accuracy_percent': relative_border_accuracy,
            'correct_border_positions': correct_border_positions
        }
    
    def draw_grid_and_score(self, canvas: np.ndarray, rows: int, cols: int, 
                            accuracy_metrics: dict) -> np.ndarray:
        """
        Draws the puzzle grid and accuracy score on the image.
        For V2, also shows rotation accuracy.
        NEW: Shows relative correct borders in green (pieces correctly connected even if rotated).
        """
        result = canvas.copy()
        h, w = canvas.shape[:2]
        piece_h = h // rows
        piece_w = w // cols
        
        # Draw correct border positions in golden first (before grid lines)
        correct_positions = accuracy_metrics.get('correct_border_positions', [])
        golden_color = (0, 215, 255)  # BGR: Golden color for correct connections
        golden_thickness = 4
        
        for border_type, r, c in correct_positions:
            if border_type == 'vertical':
                # Connection to the right
                x = (c + 1) * piece_w
                y_start = r * piece_h
                y_end = (r + 1) * piece_h
                cv2.line(result, (x, y_start), (x, y_end), golden_color, golden_thickness)
            elif border_type == 'horizontal':
                # Connection to the bottom
                y = (r + 1) * piece_h
                x_start = c * piece_w
                x_end = (c + 1) * piece_w
                cv2.line(result, (x_start, y), (x_end, y), golden_color, golden_thickness)
        
        # Golden color for grid lines (on top of green lines)
        grid_color = (0, 215, 255)  # BGR: Golden color
        grid_thickness = 2
        
        # Draw vertical lines
        for c in range(1, cols):
            x = c * piece_w
            cv2.line(result, (x, 0), (x, h), grid_color, grid_thickness)
        
        # Draw horizontal lines
        for r in range(1, rows):
            y = r * piece_h
            cv2.line(result, (0, y), (w, y), grid_color, grid_thickness)
        
        # Prepare score text
        border_acc = accuracy_metrics['border_accuracy_percent']
        correct_borders = accuracy_metrics['correct_borders']
        total_borders = accuracy_metrics['total_borders']
        rotation_acc = accuracy_metrics.get('rotation_accuracy_percent', 0)
        correct_rots = accuracy_metrics.get('correct_rotations', 0)
        total_pieces = accuracy_metrics.get('total_pieces', 0)
        relative_border_acc = accuracy_metrics.get('relative_border_accuracy_percent', 0)
        relative_correct = accuracy_metrics.get('relative_correct_borders', 0)
        
        overlay = result.copy()
        
        # Score text (three lines for V2 with relative connections)
        score_text1 = f"Correct Connections: {relative_border_acc:.1f}% ({relative_correct}/{total_borders})"
        score_text2 = f"Perfect Borders: {border_acc:.1f}% ({correct_borders}/{total_borders})"
        score_text3 = f"Rotation: {rotation_acc:.1f}% ({correct_rots}/{total_pieces})"
        
        golden_text_color = (0, 215, 255)  # Golden color for text
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate font scale
        target_width = w * 0.5
        font_scale = 1.0
        font_thickness = 2
        
        for scale in range(10, 200):
            test_scale = scale / 10.0
            test_thickness = max(2, int(test_scale * 1.5))
            (test_w, test_h), _ = cv2.getTextSize(score_text1, font, test_scale, test_thickness)
            
            if test_w >= target_width:
                font_scale = max(1.0, (scale - 1) / 10.0)
                font_thickness = max(2, int(font_scale * 1.5))
                break
            
            if scale == 199:
                font_scale = test_scale
                font_thickness = test_thickness
        
        # Calculate text sizes
        (text_w1, text_h1), _ = cv2.getTextSize(score_text1, font, font_scale, font_thickness)
        (text_w2, text_h2), _ = cv2.getTextSize(score_text2, font, font_scale, font_thickness)
        (text_w3, text_h3), _ = cv2.getTextSize(score_text3, font, font_scale, font_thickness)
        
        max_text_w = max(text_w1, text_w2, text_w3)
        total_text_h = text_h1 + text_h2 + text_h3 + 20  # 10px spacing between lines
        
        # Position for box
        box_x = (w - max_text_w) // 2 - 20
        box_y = 20
        box_w = max_text_w + 40
        box_h = total_text_h + 30
        
        # Draw semi-transparent rectangle
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
        
        # Draw text
        text_x1 = (w - text_w1) // 2
        text_y1 = box_y + text_h1 + 10
        cv2.putText(result, score_text1, (text_x1, text_y1), font, font_scale, golden_text_color, font_thickness, cv2.LINE_AA)
        
        text_x2 = (w - text_w2) // 2
        text_y2 = text_y1 + text_h2 + 10
        cv2.putText(result, score_text2, (text_x2, text_y2), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        text_x3 = (w - text_w3) // 2
        text_y3 = text_y2 + text_h3 + 10
        cv2.putText(result, score_text3, (text_x3, text_y3), font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
        
        return result
    
    def save_results(self, grid, rows, cols):
        """Saves the reconstructed image and mapping file."""
        num_slices = len(self.slices)
        image_folder = f"{self.image_name}_{num_slices}slices" if self.image_name else f"image_{num_slices}slices"
        specific_output_dir = os.path.join(self.output_dir, image_folder)
        os.makedirs(specific_output_dir, exist_ok=True)
        
        h, w = self.slices[0].image.shape[:2]
        canvas = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        
        method_name = self.__class__.__name__.replace('Solver', '').replace('V2', '').lower()
        
        map_filename = os.path.join(specific_output_dir, f"{method_name}_v2_reconstruction_map.txt")
        img_filename = os.path.join(specific_output_dir, f"{method_name}_v2_reconstructed.png")
        
        with open(map_filename, 'w') as f:
            f.write(f"[VERSION 2 - WITH ROTATION]\n")
            f.write(f"Reconstruction map for: {self.image_name or 'image'}\n")
            f.write(f"Method: {method_name.upper()} V2\n")
            f.write(f"Total pieces: {num_slices}\n")
            f.write(f"Dimensions: {rows}x{cols} pieces\n")
            f.write(f"Generated by: {self.__class__.__name__}\n")
            f.write("-" * 60 + "\n")
            f.write("POSITION | ORIGINAL FILE        | ROTATION\n")
            f.write("-" * 60 + "\n")
            
            for r in range(rows):
                for c in range(cols):
                    slc = grid[r][c]
                    f.write(f"({r},{c})    | {slc.filename:20s} | {slc.current_rotation:3d}°\n")
                    # Use the image property which applies rotation
                    canvas[r*h:(r+1)*h, c*w:(c+1)*w] = slc.image
        
        # Calculate accuracy metrics
        accuracy_metrics = self.calculate_accuracy(grid, rows, cols)
        
        # Draw grid and score
        canvas_with_overlay = self.draw_grid_and_score(canvas, rows, cols, accuracy_metrics)
        
        cv2.imwrite(img_filename, canvas_with_overlay)
        
        print(f"✓ Image saved: {img_filename}")
        print(f"✓ Map saved: {map_filename}")
        print(f"✓ Correct Connections (green): {accuracy_metrics['relative_border_accuracy_percent']:.1f}%")
        print(f"✓ Perfect borders: {accuracy_metrics['border_accuracy_percent']:.1f}%")
        print(f"✓ Rotation accuracy: {accuracy_metrics['rotation_accuracy_percent']:.1f}%")
