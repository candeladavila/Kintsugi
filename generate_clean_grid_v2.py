#!/usr/bin/env python3
"""
Generate Clean Grid Visualizations - Version 2

Creates puzzle reconstructions with only the grid overlay (no metrics text)
for color, gradient, paikin, and random methods using Version 2.
"""

import cv2
import numpy as np
import os
from pathlib import Path


def read_reconstruction_map(map_file: str) -> dict:
    """
    Read reconstruction map from text file.
    
    Args:
        map_file: Path to the reconstruction map file
        
    Returns:
        Dictionary with reconstruction info: {(row, col): (filename, rotation)}
    """
    reconstruction = {}
    rows = 0
    cols = 0
    
    with open(map_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the position mapping section (support both V1 and V2 formats)
    in_mapping = False
    for line in lines:
        # Check for header lines
        if "POSITION | ORIGINAL FILE" in line or "POSICIÓN | ARCHIVO ORIGINAL" in line:
            in_mapping = True
            continue
        
        if "----" in line:
            continue
        
        # V2 format: (0,0)    | pinguino_slice_000.png |   0°
        if in_mapping and "(" in line and ")" in line and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                pos_str = parts[0].strip()
                filename = parts[1].strip()
                rotation = 0
                
                # Extract rotation if available (V2)
                if len(parts) >= 3:
                    rot_str = parts[2].strip().replace("°", "")
                    try:
                        rotation = int(rot_str)
                    except:
                        rotation = 0
                
                # Extract row, col from (row,col)
                pos_str = pos_str.replace("(", "").replace(")", "")
                row, col = map(int, pos_str.split(","))
                
                reconstruction[(row, col)] = (filename, rotation)
                rows = max(rows, row + 1)
                cols = max(cols, col + 1)
        
        # V1 format: (0,0) -> pinguino_slice_000.png
        elif in_mapping and "(" in line and ")" in line and "->" in line:
            parts = line.strip().split("->")
            if len(parts) == 2:
                pos_str = parts[0].strip()
                filename = parts[1].strip()
                
                # Extract row, col from (row,col)
                pos_str = pos_str.replace("(", "").replace(")", "")
                row, col = map(int, pos_str.split(","))
                
                reconstruction[(row, col)] = (filename, 0)
                rows = max(rows, row + 1)
                cols = max(cols, col + 1)
    
    return reconstruction, rows, cols


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by specified angle (0, 90, 180, 270)."""
    angle = angle % 360
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def create_clean_grid(sliced_dir: str, reconstruction_map: dict, rows: int, cols: int, 
                      output_path: str):
    """
    Create a clean grid visualization without metrics text.
    
    Args:
        sliced_dir: Directory containing the sliced images
        reconstruction_map: Dictionary mapping (row, col) to (filename, rotation)
        rows: Number of rows
        cols: Number of columns
        output_path: Path to save the clean grid image
    """
    # Load first image to get dimensions
    first_file, first_rot = reconstruction_map[(0, 0)]
    first_path = os.path.join(sliced_dir, first_file)
    first_img = cv2.imread(first_path)
    
    if first_img is None:
        print(f"Error: Could not load {first_path}")
        return
    
    h, w = first_img.shape[:2]
    
    # Create canvas
    canvas = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    # Fill canvas with pieces
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in reconstruction_map:
                continue
            
            filename, rotation = reconstruction_map[(r, c)]
            img_path = os.path.join(sliced_dir, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not load {img_path}")
                continue
            
            # Apply rotation if needed
            img_rotated = rotate_image(img, rotation)
            
            canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img_rotated
    
    # Draw grid lines only
    canvas_h, canvas_w = canvas.shape[:2]
    piece_h = canvas_h // rows
    piece_w = canvas_w // cols
    
    # Gold color for grid lines
    grid_color = (0, 215, 255)
    grid_thickness = 2
    
    # Draw vertical lines
    for c_line in range(1, cols):
        x = c_line * piece_w
        cv2.line(canvas, (x, 0), (x, canvas_h), grid_color, grid_thickness)
    
    # Draw horizontal lines
    for r_line in range(1, rows):
        y = r_line * piece_h
        cv2.line(canvas, (0, y), (canvas_w, y), grid_color, grid_thickness)
    
    # Save image
    cv2.imwrite(output_path, canvas)
    print(f"✓ Clean grid saved: {output_path}")


def main():
    """Main function to generate clean grids for all methods."""
    
    # Configuration
    IMAGE_NAME = "pinguino"
    NUM_SLICES = 9
    VERSION = "ver_2"
    
    BASE_DIR = os.path.join("output_images", VERSION, f"{IMAGE_NAME}_{NUM_SLICES}slices")
    SLICED_DIR = os.path.join("sliced_images_v2", f"{IMAGE_NAME}_{NUM_SLICES}slices")
    OUTPUT_DIR = os.path.join("output_images", f"{VERSION}_clean_grid", f"{IMAGE_NAME}_{NUM_SLICES}slices")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Methods to process
    methods = ['color', 'gradient', 'paikin', 'random']
    
    print(f"{'='*60}")
    print(f"CLEAN GRID GENERATION - VERSION 2")
    print(f"{'='*60}")
    print(f"Image: {IMAGE_NAME}")
    print(f"Number of slices: {NUM_SLICES}")
    print(f"Version: {VERSION}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    for method in methods:
        print(f"\nProcessing method: {method.upper()}")
        print("-" * 40)
        
        # Read reconstruction map (V2 has _v2 suffix)
        map_file = os.path.join(BASE_DIR, f"{method}_v2_reconstruction_map.txt")
        
        if not os.path.exists(map_file):
            print(f"⚠ Warning: Map file not found: {map_file}")
            continue
        
        try:
            reconstruction_map, rows, cols = read_reconstruction_map(map_file)
            print(f"Loaded reconstruction: {rows}x{cols} grid")
            
            # Generate clean grid
            output_path = os.path.join(OUTPUT_DIR, f"{method}_clean_grid.png")
            create_clean_grid(SLICED_DIR, reconstruction_map, rows, cols, output_path)
            
        except Exception as e:
            print(f"✗ Error processing {method}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ All clean grids generated successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
