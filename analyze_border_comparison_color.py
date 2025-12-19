#!/usr/bin/env python3
"""
Border Comparison Analysis Tool

Analyzes and visualizes border compatibility between puzzle pieces
using the color-based method from puzzle_reconstructor/color_reconstructor.py

This script compares:
1. Right border of slice_001 with left border of slice_002 (correct match)
2. Right border of slice_001 with left border of slice_006 (incorrect match)

Generates matplotlib visualizations showing both borders and compatibility scores.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def extract_lab_border(img: np.ndarray, side: str, border_width: int = 100) -> np.ndarray:
    """
    Extract a border from an image in LAB color space.
    
    Args:
        img: Input image in BGR format
        side: 'top', 'bottom', 'left', 'right'
        border_width: Width of border to extract in pixels
        
    Returns:
        Border region in LAB color space as float32
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    if side == 'top':
        return lab[0:border_width, :, :]
    elif side == 'bottom':
        return lab[-border_width:, :, :]
    elif side == 'left':
        return lab[:, 0:border_width, :]
    elif side == 'right':
        return lab[:, -border_width:, :]
    else:
        raise ValueError(f"Invalid side: {side}")


def calculate_border_compatibility(border_a: np.ndarray, border_b: np.ndarray) -> tuple:
    """
    Calculate compatibility score between two borders using the same method
    as puzzle_reconstructor/color_reconstructor.py
    
    Args:
        border_a: Right border of piece A (LAB format, shape: height, width, 3)
        border_b: Left border of piece B (LAB format, shape: height, width, 3)
        
    Returns:
        Tuple of (compatibility_score, edge_a, edge_b, diff_per_pixel)
        - compatibility_score: Average Euclidean distance (lower = better match)
        - edge_a: Right edge pixels from border A (height, 3)
        - edge_b: Left edge pixels from border B (height, 3)
        - diff_per_pixel: Distance for each pixel pair (height,)
    """
    # Extract the contact edges (last column of A, first column of B)
    edge_a = border_a[:, -1, :]  # (height, 3)
    edge_b = border_b[:, 0, :]   # (height, 3)
    
    # Euclidean distance between the color vectors (L, a, b)
    diff_per_pixel = np.linalg.norm(edge_a - edge_b, axis=1)  # (height,)
    compatibility_score = np.mean(diff_per_pixel)
    
    return compatibility_score, edge_a, edge_b, diff_per_pixel


def visualize_border_comparison(img_a, img_b, border_a, border_b, 
                                  edge_a, edge_b, diff_per_pixel,
                                  compatibility_score, slice_a_name, slice_b_name,
                                  output_path, border_width=100):
    """
    Create a simplified visualization of border comparison.
    
    Args:
        img_a: Full image of piece A (BGR)
        img_b: Full image of piece B (BGR)
        border_a: Right border of A (LAB)
        border_b: Left border of B (LAB)
        edge_a: Contact edge of A (LAB, shape: height, 3)
        edge_b: Contact edge of B (LAB, shape: height, 3)
        diff_per_pixel: Difference per pixel (height,)
        compatibility_score: Overall compatibility score
        slice_a_name: Name of slice A
        slice_b_name: Name of slice B
        output_path: Path to save the visualization
        border_width: Width of border region
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Title with overall compatibility score
    fig.suptitle(f'Border Comparison: {slice_a_name} (right) vs {slice_b_name} (left)\n'
                 f'Overall Compatibility Score: {compatibility_score:.2f} (lower = better match)',
                 fontsize=14, fontweight='bold')
    
    # Row 1: Full pieces with border highlights
    ax1 = fig.add_subplot(gs[0, 0])
    img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    h, w = img_a.shape[:2]
    # Draw red rectangle on right border
    img_a_display = img_a_rgb.copy()
    cv2.rectangle(img_a_display, (w-border_width, 0), (w, h), (255, 0, 0), 3)
    ax1.imshow(img_a_display)
    ax1.set_title(f'{slice_a_name}\n(Red = right border region)', fontsize=11)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
    h, w = img_b.shape[:2]
    # Draw blue rectangle on left border
    img_b_display = img_b_rgb.copy()
    cv2.rectangle(img_b_display, (0, 0), (border_width, h), (0, 0, 255), 3)
    ax2.imshow(img_b_display)
    ax2.set_title(f'{slice_b_name}\n(Blue = left border region)', fontsize=11)
    ax2.axis('off')
    
    # Row 2: Extracted borders
    ax3 = fig.add_subplot(gs[1, 0])
    # Convert LAB border back to RGB for visualization
    border_a_bgr = cv2.cvtColor(border_a.astype(np.uint8), cv2.COLOR_LAB2BGR)
    border_a_rgb = cv2.cvtColor(border_a_bgr, cv2.COLOR_BGR2RGB)
    ax3.imshow(border_a_rgb)
    ax3.set_title(f'Right Border of {slice_a_name}\n({border_width}px width)', fontsize=10)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 1])
    border_b_bgr = cv2.cvtColor(border_b.astype(np.uint8), cv2.COLOR_LAB2BGR)
    border_b_rgb = cv2.cvtColor(border_b_bgr, cv2.COLOR_BGR2RGB)
    ax4.imshow(border_b_rgb)
    ax4.set_title(f'Left Border of {slice_b_name}\n({border_width}px width)', fontsize=10)
    ax4.axis('off')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    plt.close()


def main():
    """Main analysis function."""
    
    # Configuration
    IMAGE_NAME = "pinguino"
    NUM_SLICES = 9
    SLICED_DIR = "sliced_images_v1"
    OUTPUT_DIR = "border_analysis_output"
    BORDER_WIDTH = 100
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Construct paths
    sliced_folder = os.path.join(SLICED_DIR, f"{IMAGE_NAME}_{NUM_SLICES}slices")
    
    print(f"{'='*60}")
    print(f"BORDER COMPARISON ANALYSIS - COLOR METHOD")
    print(f"{'='*60}")
    print(f"Image: {IMAGE_NAME}")
    print(f"Number of slices: {NUM_SLICES}")
    print(f"Sliced folder: {sliced_folder}")
    print(f"Border width: {BORDER_WIDTH}px")
    print(f"{'='*60}\n")
    
    # Define comparisons to perform
    comparisons = [
        {
            'slice_a': f"{IMAGE_NAME}_slice_001.png",
            'slice_b': f"{IMAGE_NAME}_slice_002.png",
            'description': 'CORRECT MATCH (adjacent pieces)',
            'output_file': 'comparison_001_vs_002_correct.png'
        },
        {
            'slice_a': f"{IMAGE_NAME}_slice_001.png",
            'slice_b': f"{IMAGE_NAME}_slice_006.png",
            'description': 'INCORRECT MATCH (non-adjacent pieces)',
            'output_file': 'comparison_001_vs_006_incorrect.png'
        }
    ]
    
    results = []
    
    for idx, comparison in enumerate(comparisons, 1):
        print(f"\n{'='*60}")
        print(f"COMPARISON {idx}/2: {comparison['description']}")
        print(f"{'='*60}")
        
        # Load images
        slice_a_path = os.path.join(sliced_folder, comparison['slice_a'])
        slice_b_path = os.path.join(sliced_folder, comparison['slice_b'])
        
        if not os.path.exists(slice_a_path):
            print(f"Error: File not found: {slice_a_path}")
            continue
        if not os.path.exists(slice_b_path):
            print(f"Error: File not found: {slice_b_path}")
            continue
        
        img_a = cv2.imread(slice_a_path)
        img_b = cv2.imread(slice_b_path)
        
        print(f"Loading: {comparison['slice_a']}")
        print(f"Loading: {comparison['slice_b']}")
        
        # Extract borders in LAB color space
        border_a_right = extract_lab_border(img_a, 'right', BORDER_WIDTH)
        border_b_left = extract_lab_border(img_b, 'left', BORDER_WIDTH)
        
        print(f"Border A (right) shape: {border_a_right.shape}")
        print(f"Border B (left) shape: {border_b_left.shape}")
        
        # Calculate compatibility
        score, edge_a, edge_b, diff_per_pixel = calculate_border_compatibility(
            border_a_right, border_b_left
        )
        
        print(f"\n📊 RESULTS:")
        print(f"  Compatibility Score: {score:.2f}")
        print(f"  Min difference: {np.min(diff_per_pixel):.2f}")
        print(f"  Max difference: {np.max(diff_per_pixel):.2f}")
        print(f"  Std deviation: {np.std(diff_per_pixel):.2f}")
        
        results.append({
            'comparison': comparison['description'],
            'score': score,
            'slice_a': comparison['slice_a'],
            'slice_b': comparison['slice_b']
        })
        
        # Create visualization
        output_path = os.path.join(OUTPUT_DIR, comparison['output_file'])
        visualize_border_comparison(
            img_a, img_b,
            border_a_right, border_b_left,
            edge_a, edge_b, diff_per_pixel,
            score,
            comparison['slice_a'],
            comparison['slice_b'],
            output_path,
            BORDER_WIDTH
        )
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for result in results:
        print(f"\n{result['comparison']}")
        print(f"  {result['slice_a']} vs {result['slice_b']}")
        print(f"  Score: {result['score']:.2f}")
    
    if len(results) == 2:
        score_diff = abs(results[0]['score'] - results[1]['score'])
        print(f"\nScore Difference: {score_diff:.2f}")
        print(f"The correct match has a {'LOWER' if results[0]['score'] < results[1]['score'] else 'HIGHER'} score")
    
    print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
