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
    # Features (borders) for analysis
    borders: Dict[str, np.ndarray]


class PuzzleSolverBase:
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = "", border_width: int = 10):
        self.sliced_dir = sliced_dir
        self.output_dir = output_dir
        self.image_name = image_name
        self.slices: List[ImageSlice] = []
        self.border_width = border_width  # Border width to analyze (configurable, default 10 pixels)

    def extract_features(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Extracts borders for analysis. Can be overridden."""
        w_b = self.border_width
        return {
            'top': img[0:w_b, :],
            'bottom': img[-w_b:, :],
            'left': img[:, 0:w_b],
            'right': img[:, -w_b:]
        }

    def load_slices(self, original_name_pattern: str):
        """Load the existing slices from the specified folder."""
        # Search files with pattern: name_slice_XXX.png in the specified folder
        search_pattern = os.path.join(self.sliced_dir, f"{original_name_pattern}_slice_*.png")

        # Sort numerically to process in order
        try:
            files = sorted(glob.glob(search_pattern), key=lambda x: int(x.split('_slice_')[1].split('.')[0]))
        except IndexError:
            # Fallback if the name does not follow the exact format
            files = sorted(glob.glob(search_pattern))

        if not files:
            raise FileNotFoundError(f"No images found in: {search_pattern}")

        print(f"[{self.__class__.__name__}] Loading {len(files)} slices from {self.sliced_dir}...")

        for idx, fpath in enumerate(files):
            img = cv2.imread(fpath)
            if img is None:
                continue

            features = self.extract_features(img)
            self.slices.append(ImageSlice(idx, os.path.basename(fpath), img, features))

    def calculate_cost(self, idx_a: int, idx_b: int, direction: str) -> float:
        """Must be implemented by subclasses (Gradient and Color)."""
        raise NotImplementedError

    def find_top_left_corner(self, n_slices: int) -> int:
        """
        Finds the piece that has the worst best-match ABOVE and to the LEFT.
        That piece is likely the top-left corner.

        Strategy:
        - For each piece i, compute:
            min_left = min_j cost(j -> i, horizontal)  (some piece j would be to the LEFT of i)
            min_top  = min_j cost(j -> i, vertical)    (some piece j would be ABOVE i)
        - Choose the i that maximizes (min_left + min_top).
            Higher means: even its best possible neighbor on those sides is still a poor match.
        """
        max_min_cost = -1.0
        best_candidate = 0

        for i in range(n_slices):
            min_left = float('inf')
            min_top = float('inf')

            for j in range(n_slices):
                if i == j:
                    continue

                c_left = self.calculate_cost(j, i, 'horizontal')
                if c_left < min_left:
                    min_left = c_left

                c_top = self.calculate_cost(j, i, 'vertical')
                if c_top < min_top:
                    min_top = c_top

            corner_score = min_left + min_top
            if corner_score > max_min_cost:
                max_min_cost = corner_score
                best_candidate = i

        return best_candidate

    def solve(self):
        """Automatic Greedy algorithm to reconstruct the puzzle."""
        n_slices = len(self.slices)
        side = int(math.sqrt(n_slices))
        rows, cols = side, side

        # If not a perfect square, try to adjust (e.g. 2x3 for 6 pieces)
        if rows * cols != n_slices:
            # Simple logic: if not square, assume it's wide
            # This can be improved if you know the dimensions
            pass

        grid = [[None for _ in range(cols)] for _ in range(rows)]
        used_indices = set()

        # 1. Detect top-left corner
        print("Searching for top-left corner...")
        start_idx = self.find_top_left_corner(n_slices)
        grid[0][0] = self.slices[start_idx]
        used_indices.add(start_idx)
        print(f"-> Selected start piece: {self.slices[start_idx].filename}")

        # 2. Fill grid
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0:
                    continue

                best_idx = -1
                min_cost = float('inf')

                for idx in range(n_slices):
                    if idx in used_indices:
                        continue

                    cost = 0.0
                    count = 0

                    if c > 0:  # Compare with left neighbor
                        left_slice = grid[r][c - 1]
                        left_idx = self.slices.index(left_slice)  # robust: do not rely on piece.id
                        cost += self.calculate_cost(left_idx, idx, 'horizontal')
                        count += 1

                    if r > 0:  # Compare with top neighbor
                        top_slice = grid[r - 1][c]
                        top_idx = self.slices.index(top_slice)  # robust: do not rely on piece.id
                        cost += self.calculate_cost(top_idx, idx, 'vertical')
                        count += 1

                    avg_cost = cost / count if count > 0 else float('inf')

                    if avg_cost < min_cost:
                        min_cost = avg_cost
                        best_idx = idx

                # Safety fallback
                if best_idx == -1:
                    best_idx = next(i for i in range(n_slices) if i not in used_indices)

                grid[r][c] = self.slices[best_idx]
                used_indices.add(best_idx)

        self.save_results(grid, rows, cols)

    def load_solution_mapping(self) -> dict:
        """
        Load the correct solution mapping from the _order.txt file

        Returns:
            Dictionary {filename: (correct_row, correct_col)}
        """
        order_file = os.path.join(self.sliced_dir, f"{self.image_name}_order.txt")
        solution_map = {}

        if not os.path.exists(order_file):
            print(f"⚠ Warning: solution file not found: {order_file}")
            return solution_map

        try:
            with open(order_file, 'r') as f:
                lines = f.readlines()

            # Find section "ORDEN CORRECTO PARA RECOMPOSICIÓN"
            in_section = False
            for line in lines:
                if "ORDEN CORRECTO PARA RECOMPOSICIÓN" in line:
                    in_section = True
                    continue

                if in_section and "|" in line and "_slice_" in line:
                    parts = line.split("|")
                    if len(parts) >= 4:
                        filename = parts[1].strip()
                        row = int(parts[2].strip())
                        col = int(parts[3].strip())
                        solution_map[filename] = (row, col)

        except Exception as e:
            print(f"⚠ Warning: error reading solution file: {e}")

        return solution_map

    def calculate_accuracy(self, grid, rows: int, cols: int) -> dict:
        """
        Calculate reconstruction accuracy based on correctly connected borders.
        A border is correct if the two adjacent pieces should be together in the solution.

        Returns:
            Dictionary with accuracy metrics
        """
        solution_map = self.load_solution_mapping()

        if not solution_map:
            return {
                'correct_borders': 0,
                'total_borders': 0,
                'border_accuracy_percent': 0.0
            }

        # Create inverse mapping: (correct_row, correct_col) -> filename
        position_to_file = {}
        for filename, (row, col) in solution_map.items():
            position_to_file[(row, col)] = filename

        correct_borders = 0
        total_borders = 0

        # Verify correct borders (adjacencies)
        for r in range(rows):
            for c in range(cols):
                piece = grid[r][c]
                if piece.filename not in solution_map:
                    continue

                piece_correct_row, piece_correct_col = solution_map[piece.filename]

                # Verify right neighbor
                if c < cols - 1:
                    total_borders += 1
                    right_piece = grid[r][c + 1]

                    expected_right_pos = (piece_correct_row, piece_correct_col + 1)
                    if expected_right_pos in position_to_file:
                        expected_right_file = position_to_file[expected_right_pos]
                        if right_piece.filename == expected_right_file:
                            correct_borders += 1

                # Verify bottom neighbor
                if r < rows - 1:
                    total_borders += 1
                    bottom_piece = grid[r + 1][c]

                    expected_bottom_pos = (piece_correct_row + 1, piece_correct_col)
                    if expected_bottom_pos in position_to_file:
                        expected_bottom_file = position_to_file[expected_bottom_pos]
                        if bottom_piece.filename == expected_bottom_file:
                            correct_borders += 1

        border_accuracy = (correct_borders / total_borders * 100) if total_borders > 0 else 0.0

        return {
            'correct_borders': correct_borders,
            'total_borders': total_borders,
            'border_accuracy_percent': border_accuracy
        }

    def draw_grid_and_score(self, canvas: np.ndarray, rows: int, cols: int,
                            accuracy_metrics: dict) -> np.ndarray:
        """
        Draw the puzzle grid and the accuracy score on the image.

        Args:
            canvas: Reconstructed image
            rows: Number of rows
            cols: Number of columns
            accuracy_metrics: Accuracy metrics

        Returns:
            Image with grid and score drawn
        """
        result = canvas.copy()
        h, w = canvas.shape[:2]
        piece_h = h // rows
        piece_w = w // cols

        # Gold color for grid lines
        grid_color = (0, 215, 255)
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

        overlay = result.copy()

        # Score position (top center)
        score_text = f"Borders: {border_acc:.1f}% ({correct_borders}/{total_borders})"

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Calculate font_scale dynamically so text occupies approximately 50% of width
        target_width = w * 0.5
        font_scale = 1.0
        font_thickness = 2

        for scale in range(10, 200):  # Try from 1.0 to 20.0 in 0.1 steps
            test_scale = scale / 10.0
            test_thickness = max(2, int(test_scale * 1.5))
            (test_w, test_h), _ = cv2.getTextSize(score_text, font, test_scale, test_thickness)

            if test_w >= target_width:
                font_scale = max(1.0, (scale - 1) / 10.0)
                font_thickness = max(2, int(font_scale * 1.5))
                break

            if scale == 199:
                font_scale = test_scale
                font_thickness = test_thickness

        (text_w, text_h), _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)

        # Centered position at top
        box_x = (w - text_w) // 2 - 20
        box_y = 20
        box_w = text_w + 40
        box_h = text_h + 20

        # Draw semi-transparent black rectangle
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

        # Write text
        text_x = (w - text_w) // 2
        text_y = box_y + text_h + 10
        cv2.putText(result, score_text, (text_x, text_y), font, font_scale,
                    (255, 255, 255), font_thickness, cv2.LINE_AA)

        return result

    def save_results(self, grid, rows, cols):
        # Create specific folder for this image and number of slices
        num_slices = len(self.slices)
        image_folder = f"{self.image_name}_{num_slices}slices" if self.image_name else f"imagen_{num_slices}slices"
        specific_output_dir = os.path.join(self.output_dir, image_folder)
        os.makedirs(specific_output_dir, exist_ok=True)

        h, w = self.slices[0].image.shape[:2]
        canvas = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

        # Generate simpler filenames since they're in specific folder
        method_name = self.__class__.__name__.replace('Solver', '').lower()

        map_filename = os.path.join(specific_output_dir, f"{method_name}_reconstruction_map.txt")
        img_filename = os.path.join(specific_output_dir, f"{method_name}_reconstructed.png")

        # Build canvas and calculate accuracy
        with open(map_filename, 'w') as f:
            f.write(f"Reconstruction map for: {self.image_name or 'image'}\n")
            f.write(f"Method used: {method_name.upper()}\n")
            f.write(f"Total number of slices: {num_slices}\n")
            f.write(f"Dimensions: {rows}x{cols} slices\n")
            f.write(f"Generated by: {self.__class__.__name__}\n")
            f.write("-" * 50 + "\n")
            f.write("POSITION | ORIGINAL FILE\n")
            f.write("-" * 30 + "\n")

            for r in range(rows):
                for c in range(cols):
                    slc = grid[r][c]
                    f.write(f"({r},{c}) -> {slc.filename}\n")
                    canvas[r*h:(r+1)*h, c*w:(c+1)*w] = slc.image

        # Calculate accuracy metrics
        accuracy_metrics = self.calculate_accuracy(grid, rows, cols)

        # Draw grid and score on image
        canvas_with_overlay = self.draw_grid_and_score(canvas, rows, cols, accuracy_metrics)

        # Save image with overlay
        cv2.imwrite(img_filename, canvas_with_overlay)

        # Print only save confirmation
        print(f"✓ Image saved: {img_filename}")
        print(f"✓ Map saved: {map_filename}")
