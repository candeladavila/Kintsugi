import os
import glob
import math
import cv2
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class ImageSlice:
    id: int
    filename: str
    original_image: np.ndarray
    borders: Dict[str, np.ndarray]

class PuzzleSolverBase:
    def __init__(self, sliced_dir: str, output_dir: str, image_name: str = "", border_width: int = 10):
        self.sliced_dir = sliced_dir
        self.output_dir = output_dir
        self.image_name = image_name
        self.slices: List[ImageSlice] = []
        self.border_width = border_width

    def extract_features(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        w_b = self.border_width
        return {
            'top': img[0:w_b, :],
            'bottom': img[-w_b:, :],
            'left': img[:, 0:w_b],
            'right': img[:, -w_b:]
        }

    def load_slices(self, original_name_pattern: str):
        search_pattern = os.path.join(self.sliced_dir, f"{original_name_pattern}_slice_*.png")
        try:
            files = sorted(glob.glob(search_pattern), key=lambda x: int(x.split('_slice_')[1].split('.')[0]))
        except (IndexError, ValueError):
            files = sorted(glob.glob(search_pattern))

        if not files:
            raise FileNotFoundError(f"No images found in: {search_pattern}")

        self.slices = []
        for idx, fpath in enumerate(files):
            img = cv2.imread(fpath)
            if img is None: continue
            features = self.extract_features(img)
            self.slices.append(ImageSlice(idx, os.path.basename(fpath), img, features))
        
        print(f"✓ Loaded {len(self.slices)} slices")

    def calculate_cost(self, idx_a: int, idx_b: int, direction: str) -> float:
        raise NotImplementedError

    def find_top_left_corner(self, n_slices: int) -> int:
        max_min_cost = -1.0
        best_candidate = 0
        for i in range(n_slices):
            min_left = min([self.calculate_cost(j, i, 'horizontal') for j in range(n_slices) if i != j], default=float('inf'))
            min_top = min([self.calculate_cost(j, i, 'vertical') for j in range(n_slices) if i != j], default=float('inf'))
            if (min_left + min_top) > max_min_cost:
                max_min_cost = min_left + min_top
                best_candidate = i
        return best_candidate

    def solve(self):
        """
        Improved reconstruction using a Priority Queue (Best-First Search) 
        and MST as a structural guide.
        """
        import heapq
        
        n_slices = len(self.slices)
        side = int(math.sqrt(n_slices))
        rows, cols = side, side
        
        # 1. Graph Construction and MST
        # We build a global graph of all possible connections to identify 
        # the strongest structural links across the entire image.
        G = nx.Graph()
        for i in range(n_slices):
            for j in range(i + 1, n_slices):
                cost_h = min(self.calculate_cost(i, j, 'horizontal'), self.calculate_cost(j, i, 'horizontal'))
                cost_v = min(self.calculate_cost(i, j, 'vertical'), self.calculate_cost(j, i, 'vertical'))
                G.add_edge(i, j, weight=min(cost_h, cost_v))
        
        mst = nx.minimum_spanning_tree(G)
        
        # 2. Initialization
        # We use a dictionary for the grid to allow non-sequential filling.
        grid = {} 
        used_indices = set()
        priority_queue = [] # Format: (cost, row, col, piece_index)

        # 3. Place Seed Piece (Top-Left Corner)
        start_idx = self.find_top_left_corner(n_slices)
        grid[(0, 0)] = self.slices[start_idx]
        used_indices.add(start_idx)

        def add_neighbors_to_queue(r, c):
            """Identifies empty adjacent slots and finds the best candidates for them."""
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                # Ensure we stay within puzzle dimensions and target empty slots
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in grid:
                    self._push_best_for_slot(nr, nc, grid, used_indices, priority_queue, mst)

        # Start by looking for pieces that fit next to the corner
        add_neighbors_to_queue(0, 0)

        # 4. Main Reconstruction Loop (Best-Fit First)
        while len(used_indices) < n_slices and priority_queue:
            cost, r, c, idx = heapq.heappop(priority_queue)

            # Skip if the slot was filled or the piece was used elsewhere while in queue
            if (r, c) in grid or idx in used_indices:
                if (r, c) not in grid:
                    # If slot is still empty, find a new best candidate from remaining pieces
                    self._push_best_for_slot(r, c, grid, used_indices, priority_queue, mst)
                continue

            # Place the piece in the grid
            grid[(r, c)] = self.slices[idx]
            used_indices.add(idx)
            
            # Expand search to the neighbors of the newly placed piece
            add_neighbors_to_queue(r, c)

        # 5. Final Assembly
        # Convert dictionary to the list of lists format required by save_results
        final_grid = [[grid.get((r, c), self.slices[0]) for c in range(cols)] for r in range(rows)]
        self.save_results(final_grid, rows, cols)

    def _push_best_for_slot(self, r, c, grid, used_indices, pq, mst):
        """
        Evaluates all available pieces for a specific (r, c) slot.
        Considers all existing neighbors (up, down, left, right) for a multilateral fit.
        """
        import heapq
        best_idx, min_cost = -1, float('inf')
        
        for idx in range(len(self.slices)):
            if idx in used_indices: continue
            
            total_cost, count, bonus = 0.0, 0, 1.0
            
            # Adjacency checks: (neighbor_row, neighbor_col, direction, is_neighbor_on_top_or_left)
            adjacents = [
                (r, c-1, 'horizontal', True),  # Left neighbor (neighbor -> current)
                (r, c+1, 'horizontal', False), # Right neighbor (current -> neighbor)
                (r-1, c, 'vertical', True),    # Top neighbor (neighbor -> current)
                (r+1, c, 'vertical', False)    # Bottom neighbor (current -> neighbor)
            ]
            
            for nr, nc, direct, is_predecessor in adjacents:
                if (nr, nc) in grid:
                    neighbor_slice = grid[(nr, nc)]
                    # Get index of the neighbor piece
                    neighbor_idx = next(i for i, s in enumerate(self.slices) if s.filename == neighbor_slice.filename)
                    
                    if is_predecessor:
                        total_cost += self.calculate_cost(neighbor_idx, idx, direct)
                        if mst.has_edge(neighbor_idx, idx): bonus = 0.5
                    else:
                        total_cost += self.calculate_cost(idx, neighbor_idx, direct)
                        if mst.has_edge(idx, neighbor_idx): bonus = 0.5
                    count += 1
            
            if count > 0:
                # Average the cost if multiple neighbors exist (e.g., an inside corner)
                avg_cost = (total_cost / count) * bonus
                if avg_cost < min_cost:
                    min_cost, best_idx = avg_cost, idx
        
        if best_idx != -1:
            heapq.heappush(pq, (min_cost, r, c, best_idx))

    def load_solution_mapping(self) -> dict:
        """Parses the order.txt file to get the correct position of each piece."""
        order_file = os.path.join(self.sliced_dir, f"{self.image_name}_order.txt")
        solution_map = {}
        if not os.path.exists(order_file): return solution_map

        with open(order_file, 'r') as f:
            lines = f.readlines()
            
        start_parsing = False
        for line in lines:
            # Look for the header line with the table columns
            if ("Archivo Guardado" in line or "Saved File" in line) and "Fila" in line:
                start_parsing = True
                continue
            # Look for separator line
            if start_parsing and "---" in line:
                continue
            # Parse data lines
            if start_parsing and "|" in line:
                parts = line.split("|")
                if len(parts) >= 4:
                    try:
                        filename = parts[1].strip()
                        row = int(parts[2].strip())
                        col = int(parts[3].strip())
                        solution_map[filename] = (row, col)
                    except ValueError: 
                        continue
            # Stop parsing after the table ends
            if start_parsing and line.strip().startswith("--") and len(solution_map) > 0:
                break
        return solution_map

    def calculate_accuracy(self, grid, rows: int, cols: int) -> dict:
        solution_map = self.load_solution_mapping()
        if not solution_map: return {'correct_borders': 0, 'total_borders': 0, 'border_accuracy_percent': 0.0}

        correct_borders, total_borders = 0, 0
        for r in range(rows):
            for c in range(cols):
                p = grid[r][c]
                if p.filename not in solution_map: continue
                pr, pc = solution_map[p.filename]

                if c < cols - 1: # Right neighbor
                    total_borders += 1
                    neighbor = grid[r][c+1]
                    if neighbor.filename in solution_map:
                        nr, nc = solution_map[neighbor.filename]
                        # Correct connection: neighbor is exactly one position to the right in the solution
                        if nr == pr and nc == pc + 1: 
                            correct_borders += 1

                if r < rows - 1: # Bottom neighbor
                    total_borders += 1
                    neighbor = grid[r+1][c]
                    if neighbor.filename in solution_map:
                        nr, nc = solution_map[neighbor.filename]
                        # Correct connection: neighbor is exactly one position below in the solution
                        if nr == pr + 1 and nc == pc: 
                            correct_borders += 1

        return {
            'correct_borders': correct_borders, 
            'total_borders': total_borders, 
            'border_accuracy_percent': (correct_borders / total_borders * 100) if total_borders > 0 else 0.0
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
        grid_thickness = 4

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

        # Calculate font_scale dynamically so text occupies approximately 30% of width
        target_width = w * 0.3
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

    def save_reconstruction_map(self, grid, rows: int, cols: int, out_dir: str, method_name: str):
        """Save reconstruction map to text file."""
        map_filename = f"{method_name}_reconstruction_map.txt"
        map_path = os.path.join(out_dir, map_filename)
        
        with open(map_path, 'w') as f:
            f.write("Reconstruction Map\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Method: {method_name}\n")
            f.write(f"Grid size: {rows}x{cols}\n\n")
            f.write("Position Mapping:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Position':<15} | {'Filename':<30} | {'Original Pos'}\n")
            f.write("-" * 80 + "\n")
            
            solution_map = self.load_solution_mapping()
            
            for r in range(rows):
                for c in range(cols):
                    piece = grid[r][c]
                    position_str = f"({r},{c})"
                    
                    # Get original position if available
                    if piece.filename in solution_map:
                        orig_r, orig_c = solution_map[piece.filename]
                        orig_pos_str = f"({orig_r},{orig_c})"
                    else:
                        orig_pos_str = "Unknown"
                    
                    f.write(f"{position_str:<15} | {piece.filename:<30} | {orig_pos_str}\n")
            
            f.write("\n")
        
        print(f"✓ Map saved: {map_filename}")

    def save_results(self, grid, rows, cols):
        num_slices = len(self.slices)
        out_dir = os.path.join(self.output_dir, f"{self.image_name}_{num_slices}slices")
        os.makedirs(out_dir, exist_ok=True)
        
        h, w = self.slices[0].original_image.shape[:2]
        canvas = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                canvas[r*h:(r+1)*h, c*w:(c+1)*w] = grid[r][c].original_image
        
        metrics = self.calculate_accuracy(grid, rows, cols)
        final_img = self.draw_grid_and_score(canvas, rows, cols, metrics)
        
        method_name = self.__class__.__name__.lower().replace('solver','')
        filename = f"{method_name}_reconstructed.png"
        cv2.imwrite(os.path.join(out_dir, filename), final_img)
        print(f"✓ Saved: {filename} | Borders: {metrics['border_accuracy_percent']:.1f}%")
        
        # Save reconstruction map
        self.save_reconstruction_map(grid, rows, cols, out_dir, method_name)