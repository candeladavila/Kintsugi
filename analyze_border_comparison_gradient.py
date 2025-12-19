#!/usr/bin/env python3
"""
Border Comparison Analysis Tool - GRADIENT METHOD

Analyzes and visualizes border compatibility between puzzle pieces
using the gradient-based method from puzzle_reconstructor/gradient_reconstructor.py

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


def extract_border(img: np.ndarray, side: str, border_width: int = 100) -> np.ndarray:
    """
    Extract a border from an image.
    
    Args:
        img: Input image in BGR format
        side: 'top', 'bottom', 'left', 'right'
        border_width: Width of border to extract in pixels
        
    Returns:
        Border region in BGR format
    """
    h, w = img.shape[:2]
    border_w = min(border_width, w, h)
    
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


def create_border_pair_image(border1: np.ndarray, border2: np.ndarray, orientation: str) -> np.ndarray:
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
        combined = np.hstack([border1, border2])
    elif orientation == 'vertical':
        combined = np.vstack([border1, border2])
    else:
        raise ValueError(f"Invalid orientation: {orientation}")
    
    return combined


def calculate_gradient_continuity(combined_border: np.ndarray, orientation: str) -> tuple:
    """
    Calculate gradient continuity in the junction area of two borders.
    Analyzes gradients in COLOR (each channel separately) for better accuracy.
    
    Args:
        combined_border: Image with two borders joined (BGR)
        orientation: 'horizontal' or 'vertical'
        
    Returns:
        Tuple of (quality_index, diff_grad, diff_color, smoothness_grad, smoothness_color)
    """
    # Convert to LAB for perceptual analysis
    lab = cv2.cvtColor(combined_border, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute gradients in each LAB channel
    gradients_lab = []
    for channel in range(3):
        grad_x = cv2.Sobel(lab[:, :, channel], cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(lab[:, :, channel], cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradients_lab.append(grad_mag)
    
    # Combined gradient magnitude (weighted average of LAB channels)
    grad_magnitude = (gradients_lab[0] * 0.6 + 
                     gradients_lab[1] * 0.2 + 
                     gradients_lab[2] * 0.2)
    
    # Analyze continuity at the joining line
    h, w = combined_border.shape[:2]
    
    if orientation == 'horizontal':
        center = w // 2
        region_width = 30
        left_bound = max(0, center - region_width // 2)
        right_bound = min(w, center + region_width // 2)
        
        junction_region_grad = grad_magnitude[:, left_bound:right_bound]
        junction_region_lab = lab[:, left_bound:right_bound, :]
        
        center_local = (right_bound - left_bound) // 2
        if center_local > 0 and center_local < junction_region_grad.shape[1] - 1:
            left_grad = junction_region_grad[:, center_local - 1]
            right_grad = junction_region_grad[:, center_local + 1]
            diff_grad = np.mean(np.abs(left_grad - right_grad))
            
            left_color = junction_region_lab[:, center_local - 1, :]
            right_color = junction_region_lab[:, center_local + 1, :]
            diff_color = np.mean(np.sqrt(np.sum((left_color - right_color)**2, axis=1)))
        else:
            diff_grad = np.mean(junction_region_grad)
            diff_color = 100.0
            
    elif orientation == 'vertical':
        center = h // 2
        region_height = 30
        top_bound = max(0, center - region_height // 2)
        bottom_bound = min(h, center + region_height // 2)
        
        junction_region_grad = grad_magnitude[top_bound:bottom_bound, :]
        junction_region_lab = lab[top_bound:bottom_bound, :, :]
        
        center_local = (bottom_bound - top_bound) // 2
        if center_local > 0 and center_local < junction_region_grad.shape[0] - 1:
            top_grad = junction_region_grad[center_local - 1, :]
            bottom_grad = junction_region_grad[center_local + 1, :]
            diff_grad = np.mean(np.abs(top_grad - bottom_grad))
            
            top_color = junction_region_lab[center_local - 1, :, :]
            bottom_color = junction_region_lab[center_local + 1, :, :]
            diff_color = np.mean(np.sqrt(np.sum((top_color - bottom_color)**2, axis=1)))
        else:
            diff_grad = np.mean(junction_region_grad)
            diff_color = 100.0
    else:
        diff_grad = float('inf')
        diff_color = float('inf')
    
    # Compute smoothness at the junction
    smoothness_grad = np.std(junction_region_grad)
    smoothness_color = np.std(junction_region_lab)
    
    # Combined quality index with adjusted weights
    quality_index = (0.4 * diff_grad + 
                    0.4 * diff_color + 
                    0.1 * smoothness_grad + 
                    0.1 * smoothness_color)
    
    return quality_index, diff_grad, diff_color, smoothness_grad, smoothness_color


def calculate_border_compatibility(border_a: np.ndarray, border_b: np.ndarray) -> tuple:
    """
    Calculate compatibility score between two borders using gradient method.
    
    Args:
        border_a: Right border of piece A (BGR format)
        border_b: Left border of piece B (BGR format)
        
    Returns:
        Tuple of (compatibility_score, diff_grad, diff_color, smoothness_grad, smoothness_color)
    """
    # Create combined border image
    combined = create_border_pair_image(border_a, border_b, 'horizontal')
    
    # Calculate gradient continuity
    quality, diff_grad, diff_color, smoothness_grad, smoothness_color = calculate_gradient_continuity(
        combined, 'horizontal'
    )
    
    return quality, diff_grad, diff_color, smoothness_grad, smoothness_color


def visualize_border_comparison(img_a, img_b, border_a, border_b,
                                  compatibility_score, slice_a_name, slice_b_name,
                                  output_path, border_width=100):
    """
    Create a simplified visualization of border comparison.
    
    Args:
        img_a: Full image of piece A (BGR)
        img_b: Full image of piece B (BGR)
        border_a: Right border of A (BGR)
        border_b: Left border of B (BGR)
        compatibility_score: Overall compatibility score
        slice_a_name: Name of slice A
        slice_b_name: Name of slice B
        output_path: Path to save the visualization
        border_width: Width of border region
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Title with overall compatibility score
    fig.suptitle(f'Border Comparison (GRADIENT): {slice_a_name} (right) vs {slice_b_name} (left)\n'
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
    border_a_rgb = cv2.cvtColor(border_a, cv2.COLOR_BGR2RGB)
    ax3.imshow(border_a_rgb)
    ax3.set_title(f'Right Border of {slice_a_name}\n({border_width}px width)', fontsize=10)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 1])
    border_b_rgb = cv2.cvtColor(border_b, cv2.COLOR_BGR2RGB)
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
    print(f"BORDER COMPARISON ANALYSIS - GRADIENT METHOD")
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
            'output_file': 'comparison_001_vs_002_correct_gradient.png'
        },
        {
            'slice_a': f"{IMAGE_NAME}_slice_001.png",
            'slice_b': f"{IMAGE_NAME}_slice_006.png",
            'description': 'INCORRECT MATCH (non-adjacent pieces)',
            'output_file': 'comparison_001_vs_006_incorrect_gradient.png'
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
        
        # Extract borders
        border_a_right = extract_border(img_a, 'right', BORDER_WIDTH)
        border_b_left = extract_border(img_b, 'left', BORDER_WIDTH)
        
        print(f"Border A (right) shape: {border_a_right.shape}")
        print(f"Border B (left) shape: {border_b_left.shape}")
        
        # Calculate compatibility
        score, diff_grad, diff_color, smoothness_grad, smoothness_color = calculate_border_compatibility(
            border_a_right, border_b_left
        )
        
        print(f"\n📊 RESULTS:")
        print(f"  Overall Compatibility Score: {score:.2f}")
        print(f"  Gradient difference: {diff_grad:.2f}")
        print(f"  Color difference: {diff_color:.2f}")
        print(f"  Gradient smoothness: {smoothness_grad:.2f}")
        print(f"  Color smoothness: {smoothness_color:.2f}")
        
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
