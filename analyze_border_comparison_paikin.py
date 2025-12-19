#!/usr/bin/env python3
"""
Border Comparison Analysis Tool - PAIKIN-TAL METHOD

Analyzes and visualizes border compatibility between puzzle pieces
using the Paikin-Tal method from puzzle_reconstructor/paikin_reconstructor.py

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


def extract_features(img: np.ndarray, border_width: int = 10):
    """
    Extract features for Paikin-Tal method.
    Uses LAB color space + Gradient Magnitude.
    
    Args:
        img: Input image in BGR format
        border_width: Width of border to extract in pixels
        
    Returns:
        Dictionary with features for each border side
    """
    h, w = img.shape[:2]
    
    # LAB color space normalized to [0,1]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    
    # Gradient Magnitude normalized to [0,1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    g_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad_mag = cv2.magnitude(g_x, g_y)
    grad_mag = np.clip(grad_mag, 0, 1.0)
    
    # Combine: 3 LAB channels + 1 magnitude channel = 4 channels
    combined = np.dstack([lab, grad_mag[..., None]]).astype(np.float32)
    
    w_b = min(border_width, h // 2, w // 2)
    return {
        'top': combined[0:w_b, :, :],
        'bottom': combined[-w_b:, :, :],
        'left': combined[:, 0:w_b, :],
        'right': combined[:, -w_b:, :]
    }


def calculate_cost(feats_a, feats_b, multi_border_width=3, 
                   bg_activity_threshold=0.03, weight_color=0.4, weight_gradient=0.6):
    """
    Calculate compatibility cost between two borders using Paikin-Tal method.
    
    Args:
        feats_a: Features of piece A (right border)
        feats_b: Features of piece B (left border)
        multi_border_width: Width of edge to compare
        bg_activity_threshold: Threshold to detect background
        weight_color: Weight for color difference
        weight_gradient: Weight for gradient difference
        
    Returns:
        Tuple of (cost, act_a, act_b, diff_color, diff_grad, is_bg_a, is_bg_b)
    """
    mbw = multi_border_width
    
    # Extract contact edges
    max_w = min(feats_a.shape[1], feats_b.shape[1], mbw)
    edge_a = feats_a[:, -max_w:, :]
    edge_b = feats_b[:, :max_w, :]
    
    # Flatten [N pixels, 4 channels]
    edge_a_flat = edge_a.reshape(-1, edge_a.shape[-1])
    edge_b_flat = edge_b.reshape(-1, edge_b.shape[-1])
    
    # Measure "activity" using the magnitude channel (index 3)
    act_a = np.mean(edge_a_flat[:, 3])
    act_b = np.mean(edge_b_flat[:, 3])
    
    is_bg_a = act_a < bg_activity_threshold
    is_bg_b = act_b < bg_activity_threshold
    
    # Compute base differences
    # Color difference (Channels 0,1,2 - LAB)
    diff_color = np.linalg.norm(edge_a_flat[:, :3] - edge_b_flat[:, :3], axis=1).mean()
    # Gradient difference (Channel 3 - Magnitude)
    diff_grad = np.abs(edge_a_flat[:, 3] - edge_b_flat[:, 3]).mean()
    
    cost = 0.0
    
    # CASE 1: BOTH ARE BACKGROUND
    if is_bg_a and is_bg_b:
        cost = diff_color * 2.0
        
    # CASE 2: MIXED (background with object)
    elif is_bg_a != is_bg_b:
        cost = 5.0 + diff_color + diff_grad
        
    # CASE 3: BOTH ARE OBJECTS
    else:
        cost = (weight_color * diff_color) + (weight_gradient * diff_grad)
    
    return cost, act_a, act_b, diff_color, diff_grad, is_bg_a, is_bg_b


def calculate_border_compatibility(img_a: np.ndarray, img_b: np.ndarray, 
                                   border_width: int = 10) -> tuple:
    """
    Calculate compatibility score between two images using Paikin-Tal method.
    
    Args:
        img_a: Image of piece A (BGR format)
        img_b: Image of piece B (BGR format)
        border_width: Width of border to extract
        
    Returns:
        Tuple of (cost, act_a, act_b, diff_color, diff_grad, is_bg_a, is_bg_b, feats_a_right, feats_b_left)
    """
    # Extract features
    feats_a = extract_features(img_a, border_width)
    feats_b = extract_features(img_b, border_width)
    
    # Calculate cost for horizontal comparison (right of A vs left of B)
    cost, act_a, act_b, diff_color, diff_grad, is_bg_a, is_bg_b = calculate_cost(
        feats_a['right'], feats_b['left']
    )
    
    return cost, act_a, act_b, diff_color, diff_grad, is_bg_a, is_bg_b, feats_a['right'], feats_b['left']


def visualize_border_comparison(img_a, img_b, border_a, border_b,
                                  compatibility_score, slice_a_name, slice_b_name,
                                  output_path, border_width=10):
    """
    Create a simplified visualization of border comparison.
    
    Args:
        img_a: Full image of piece A (BGR)
        img_b: Full image of piece B (BGR)
        border_a: Right border features of A (4 channels)
        border_b: Left border features of B (4 channels)
        compatibility_score: Overall compatibility score
        slice_a_name: Name of slice A
        slice_b_name: Name of slice B
        output_path: Path to save the visualization
        border_width: Width of border region
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Title with overall compatibility score
    fig.suptitle(f'Border Comparison (PAIKIN-TAL): {slice_a_name} (right) vs {slice_b_name} (left)\n'
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
    
    # Row 2: Extracted borders (LAB visualization from features)
    ax3 = fig.add_subplot(gs[1, 0])
    # Extract LAB channels from features (channels 0,1,2)
    border_a_lab = (border_a[:, :, :3] * 255).astype(np.uint8)
    border_a_bgr = cv2.cvtColor(border_a_lab, cv2.COLOR_LAB2BGR)
    border_a_rgb = cv2.cvtColor(border_a_bgr, cv2.COLOR_BGR2RGB)
    ax3.imshow(border_a_rgb)
    ax3.set_title(f'Right Border of {slice_a_name}\n({border_width}px width)', fontsize=10)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 1])
    border_b_lab = (border_b[:, :, :3] * 255).astype(np.uint8)
    border_b_bgr = cv2.cvtColor(border_b_lab, cv2.COLOR_LAB2BGR)
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
    BORDER_WIDTH = 100  # Using wider borders like other methods
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Construct paths
    sliced_folder = os.path.join(SLICED_DIR, f"{IMAGE_NAME}_{NUM_SLICES}slices")
    
    print(f"{'='*60}")
    print(f"BORDER COMPARISON ANALYSIS - PAIKIN-TAL METHOD")
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
            'output_file': 'comparison_001_vs_002_correct_paikin.png'
        },
        {
            'slice_a': f"{IMAGE_NAME}_slice_001.png",
            'slice_b': f"{IMAGE_NAME}_slice_006.png",
            'description': 'INCORRECT MATCH (non-adjacent pieces)',
            'output_file': 'comparison_001_vs_006_incorrect_paikin.png'
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
        
        # Calculate compatibility
        cost, act_a, act_b, diff_color, diff_grad, is_bg_a, is_bg_b, feats_a, feats_b = calculate_border_compatibility(
            img_a, img_b, BORDER_WIDTH
        )
        
        print(f"\n📊 RESULTS:")
        print(f"  Overall Compatibility Score: {cost:.2f}")
        print(f"  Activity A: {act_a:.4f} {'(BACKGROUND)' if is_bg_a else '(OBJECT)'}")
        print(f"  Activity B: {act_b:.4f} {'(BACKGROUND)' if is_bg_b else '(OBJECT)'}")
        print(f"  Color difference: {diff_color:.4f}")
        print(f"  Gradient difference: {diff_grad:.4f}")
        
        results.append({
            'comparison': comparison['description'],
            'score': cost,
            'slice_a': comparison['slice_a'],
            'slice_b': comparison['slice_b']
        })
        
        # Create visualization
        output_path = os.path.join(OUTPUT_DIR, comparison['output_file'])
        visualize_border_comparison(
            img_a, img_b,
            feats_a, feats_b,
            cost,
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
