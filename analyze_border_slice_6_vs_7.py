#!/usr/bin/env python3
"""
Border Comparison Analysis - Slice 6 vs Slice 7

Compara el borde INFERIOR de slice_006 con los 4 bordes de slice_007
usando el método de color del puzzle_reconstructor_v2.

Genera 4 visualizaciones:
1. Bottom of slice_6 vs TOP of slice_7
2. Bottom of slice_6 vs BOTTOM of slice_7
3. Bottom of slice_6 vs LEFT of slice_7
4. Bottom of slice_6 vs RIGHT of slice_7
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


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


def calculate_border_compatibility_horizontal(border_a: np.ndarray, border_b: np.ndarray) -> tuple:
    """
    Calculate compatibility between two horizontal borders.
    Compares them pixel by pixel along their contact edge.
    
    Args:
        border_a: Bottom border of piece A (LAB, shape: height, width, 3)
        border_b: Any border of piece B (LAB, might need adjustment)
        
    Returns:
        (score, edge_a, edge_b, diff_per_pixel)
    """
    # Para borde horizontal (bottom), usamos la última fila
    # border_a shape: (border_width, width, 3)
    # Queremos comparar la última fila con la primera fila de border_b si es top
    # O necesitamos adaptar si es left/right
    
    h_a, w_a = border_a.shape[:2]
    h_b, w_b = border_b.shape[:2]
    
    # Extraer el borde de contacto
    edge_a = border_a[-1, :, :]  # Última fila (bottom edge) - shape: (width, 3)
    
    # Dependiendo del borde B, extraemos diferente
    if h_b < w_b:  # Borde horizontal (top o bottom)
        edge_b = border_b[0, :, :]  # Primera fila - shape: (width, 3)
    else:  # Borde vertical (left o right)
        # Transponer para que coincida
        edge_b = border_b[:, 0, :] if w_b <= h_b else border_b[:, -1, :]  # shape: (height, 3)
    
    # Asegurar que tengan el mismo tamaño para comparar
    min_len = min(edge_a.shape[0], edge_b.shape[0])
    edge_a = edge_a[:min_len, :]
    edge_b = edge_b[:min_len, :]
    
    # Calcular distancia euclidiana
    diff_per_pixel = np.linalg.norm(edge_a - edge_b, axis=1)
    score = np.mean(diff_per_pixel)
    
    return score, edge_a, edge_b, diff_per_pixel


def visualize_border_comparison(img_a, img_b, border_a, border_b, 
                                  edge_a, edge_b, diff_per_pixel,
                                  compatibility_score, slice_a_name, slice_b_name,
                                  side_a, side_b, output_path, border_width=100):
    """
    Create visualization comparing two borders.
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'Border Comparison: {slice_a_name} ({side_a}) vs {slice_b_name} ({side_b})\n'
                 f'Compatibility Score: {compatibility_score:.2f} (lower = better match)',
                 fontsize=14, fontweight='bold')
    
    # Row 1: Full pieces with border highlights
    ax1 = fig.add_subplot(gs[0, 0])
    img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    h, w = img_a.shape[:2]
    img_a_display = img_a_rgb.copy()
    
    # Draw rectangle on the specified border
    if side_a == 'bottom':
        cv2.rectangle(img_a_display, (0, h-border_width), (w, h), (255, 0, 0), 3)
    elif side_a == 'top':
        cv2.rectangle(img_a_display, (0, 0), (w, border_width), (255, 0, 0), 3)
    elif side_a == 'left':
        cv2.rectangle(img_a_display, (0, 0), (border_width, h), (255, 0, 0), 3)
    elif side_a == 'right':
        cv2.rectangle(img_a_display, (w-border_width, 0), (w, h), (255, 0, 0), 3)
    
    ax1.imshow(img_a_display)
    ax1.set_title(f'{slice_a_name}\n(Red = {side_a} border region)', fontsize=11)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
    h, w = img_b.shape[:2]
    img_b_display = img_b_rgb.copy()
    
    # Draw rectangle on the specified border
    if side_b == 'bottom':
        cv2.rectangle(img_b_display, (0, h-border_width), (w, h), (0, 0, 255), 3)
    elif side_b == 'top':
        cv2.rectangle(img_b_display, (0, 0), (w, border_width), (0, 0, 255), 3)
    elif side_b == 'left':
        cv2.rectangle(img_b_display, (0, 0), (border_width, h), (0, 0, 255), 3)
    elif side_b == 'right':
        cv2.rectangle(img_b_display, (w-border_width, 0), (w, h), (0, 0, 255), 3)
    
    ax2.imshow(img_b_display)
    ax2.set_title(f'{slice_b_name}\n(Blue = {side_b} border region)', fontsize=11)
    ax2.axis('off')
    
    # Row 2: Extracted borders
    ax3 = fig.add_subplot(gs[1, 0])
    border_a_bgr = cv2.cvtColor(border_a.astype(np.uint8), cv2.COLOR_LAB2BGR)
    border_a_rgb = cv2.cvtColor(border_a_bgr, cv2.COLOR_BGR2RGB)
    ax3.imshow(border_a_rgb)
    ax3.set_title(f'{side_a.capitalize()} Border of {slice_a_name}\n({border_width}px)', fontsize=10)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 1])
    border_b_bgr = cv2.cvtColor(border_b.astype(np.uint8), cv2.COLOR_LAB2BGR)
    border_b_rgb = cv2.cvtColor(border_b_bgr, cv2.COLOR_BGR2RGB)
    ax4.imshow(border_b_rgb)
    ax4.set_title(f'{side_b.capitalize()} Border of {slice_b_name}\n({border_width}px)', fontsize=10)
    ax4.axis('off')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Main analysis function."""
    
    # Configuration
    IMAGE_NAME = "pinguino"
    NUM_SLICES = 9
    SLICED_DIR = "sliced_images_v2"
    OUTPUT_DIR = "border_analysis_output"
    BORDER_WIDTH = 100
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Construct paths
    sliced_folder = os.path.join(SLICED_DIR, f"{IMAGE_NAME}_{NUM_SLICES}slices")
    
    print(f"{'='*70}")
    print(f"BORDER COMPARISON ANALYSIS - SLICE 6 (BOTTOM) VS SLICE 7 (ALL SIDES)")
    print(f"{'='*70}")
    print(f"Image: {IMAGE_NAME}")
    print(f"Version: V2")
    print(f"Number of slices: {NUM_SLICES}")
    print(f"Sliced folder: {sliced_folder}")
    print(f"Border width: {BORDER_WIDTH}px")
    print(f"{'='*70}\n")
    
    # Load images
    slice_6_path = os.path.join(sliced_folder, f"{IMAGE_NAME}_slice_006.png")
    slice_7_path = os.path.join(sliced_folder, f"{IMAGE_NAME}_slice_007.png")
    
    if not os.path.exists(slice_6_path):
        print(f"Error: File not found: {slice_6_path}")
        return
    if not os.path.exists(slice_7_path):
        print(f"Error: File not found: {slice_7_path}")
        return
    
    img_6 = cv2.imread(slice_6_path)
    img_7 = cv2.imread(slice_7_path)
    
    print(f"Loaded: {IMAGE_NAME}_slice_006.png - Shape: {img_6.shape}")
    print(f"Loaded: {IMAGE_NAME}_slice_007.png - Shape: {img_7.shape}")
    print()
    
    # Extract bottom border of slice 6 (this will be compared against all sides of slice 7)
    border_6_bottom = extract_lab_border(img_6, 'bottom', BORDER_WIDTH)
    print(f"Extracted BOTTOM border from slice 6: shape {border_6_bottom.shape}")
    
    # Define comparisons: bottom of slice 6 vs each side of slice 7
    comparisons = [
        {
            'side_7': 'top',
            'description': 'Bottom of Slice 6 vs TOP of Slice 7',
            'output_file': f'comparison_slice6_bottom_vs_slice7_top_color.png'
        },
        {
            'side_7': 'bottom',
            'description': 'Bottom of Slice 6 vs BOTTOM of Slice 7',
            'output_file': f'comparison_slice6_bottom_vs_slice7_bottom_color.png'
        },
        {
            'side_7': 'left',
            'description': 'Bottom of Slice 6 vs LEFT of Slice 7',
            'output_file': f'comparison_slice6_bottom_vs_slice7_left_color.png'
        },
        {
            'side_7': 'right',
            'description': 'Bottom of Slice 6 vs RIGHT of Slice 7',
            'output_file': f'comparison_slice6_bottom_vs_slice7_right_color.png'
        }
    ]
    
    results = []
    
    for idx, comparison in enumerate(comparisons, 1):
        print(f"\n{'='*70}")
        print(f"COMPARISON {idx}/4: {comparison['description']}")
        print(f"{'='*70}")
        
        # Extract the corresponding border from slice 7
        side_7 = comparison['side_7']
        border_7 = extract_lab_border(img_7, side_7, BORDER_WIDTH)
        
        print(f"Border 6 (bottom) shape: {border_6_bottom.shape}")
        print(f"Border 7 ({side_7}) shape: {border_7.shape}")
        
        # Calculate compatibility
        score, edge_6, edge_7, diff_per_pixel = calculate_border_compatibility_horizontal(
            border_6_bottom, border_7
        )
        
        print(f"\n📊 RESULTS:")
        print(f"  Compatibility Score: {score:.2f}")
        print(f"  Min difference: {np.min(diff_per_pixel):.2f}")
        print(f"  Max difference: {np.max(diff_per_pixel):.2f}")
        print(f"  Std deviation: {np.std(diff_per_pixel):.2f}")
        
        results.append({
            'comparison': comparison['description'],
            'score': score,
            'side_7': side_7
        })
        
        # Create visualization
        output_path = os.path.join(OUTPUT_DIR, comparison['output_file'])
        visualize_border_comparison(
            img_6, img_7,
            border_6_bottom, border_7,
            edge_6, edge_7, diff_per_pixel,
            score,
            f"{IMAGE_NAME}_slice_006",
            f"{IMAGE_NAME}_slice_007",
            'bottom',
            side_7,
            output_path,
            BORDER_WIDTH
        )
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY - BOTTOM OF SLICE 6 VS ALL SIDES OF SLICE 7")
    print(f"{'='*70}")
    
    # Sort by score (best match first)
    results_sorted = sorted(results, key=lambda x: x['score'])
    
    for i, result in enumerate(results_sorted, 1):
        marker = " ← BEST MATCH" if i == 1 else ""
        print(f"\n{i}. {result['comparison']}")
        print(f"   Score: {result['score']:.2f}{marker}")
    
    print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
