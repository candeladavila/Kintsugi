"""
Slice Images V2 - With Rotation Support

This version adds random rotation (0°, 90°, 180°, 270°) to each piece
during the slicing process. The reconstruction algorithms must then
determine both the correct position AND rotation of each piece.
"""

import os
import math
import cv2
import numpy as np
import argparse
import random


# Rotation angles: 0, 90, 180, 270 degrees
ROTATION_ANGLES = [0, 90, 180, 270]


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotates an image by the specified angle (must be 0, 90, 180, or 270).
    
    Args:
        img: Image to rotate
        angle: Rotation angle in degrees (0, 90, 180, 270)
        
    Returns:
        Rotated image
    """
    if angle == 0:
        return img.copy()
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"Invalid rotation angle: {angle}. Must be 0, 90, 180, or 270.")


def slice_image_v2(image_path, num_slices, output_dir="sliced_images_v2"):
    """
    Divides an image into num_slices square parts with RANDOM ROTATION
    and generates a text file with the correct order and rotation for reconstruction.
    
    Args:
        image_path (str): Path to the image to divide
        num_slices (int): Number of parts (must have exact square root)
        output_dir (str): Folder where to save the parts
        
    Returns:
        List of slice information dictionaries
    """
    # Verify that num_slices has an exact square root
    sqrt_slices = int(math.sqrt(num_slices))
    if sqrt_slices * sqrt_slices != num_slices:
        raise ValueError(f"Number {num_slices} does not have an exact square root")
    
    # Create specific output directory for this image and number of slices
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    specific_output_dir = os.path.join(output_dir, f"{base_name}_{num_slices}slices")
    os.makedirs(specific_output_dir, exist_ok=True)
    
    # Open the image
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
    except Exception as e:
        raise ValueError(f"Could not open image: {e}")
    
    # Get image dimensions (OpenCV uses height, width, channels)
    img_height, img_width = img.shape[:2]
    
    # Calculate dimensions of each piece
    slice_width = img_width // sqrt_slices
    slice_height = img_height // sqrt_slices
    
    # Verify pieces are square (required for rotation)
    if slice_width != slice_height:
        print(f"⚠ Warning: Pieces are not square ({slice_width}x{slice_height})")
        print(f"  Adjusting to use minimum dimension: {min(slice_width, slice_height)}")
        slice_size = min(slice_width, slice_height)
        slice_width = slice_height = slice_size
    
    print(f"[V2 WITH ROTATION] Dividing image {image_path} into {num_slices} parts ({sqrt_slices}x{sqrt_slices})")
    print(f"Saving to folder: {specific_output_dir}")
    print(f"Original dimensions: {img_width}x{img_height}")
    print(f"Piece dimensions: {slice_width}x{slice_height}")
    print(f"Rotation: ENABLED (random 0°, 90°, 180°, 270°)")
    
    # List to store slice order
    slice_order = []
    
    # Create list of positions and shuffle randomly
    slice_positions = []
    slice_index = 0
    for row in range(sqrt_slices):
        for col in range(sqrt_slices):
            slice_positions.append((row, col, slice_index))
            slice_index += 1
    
    # Shuffle positions randomly
    random.shuffle(slice_positions)
    
    # Process each piece in random order
    for idx, (row, col, original_position) in enumerate(slice_positions):
        # Calculate piece coordinates
        left = col * slice_width
        top = row * slice_height
        right = left + slice_width
        bottom = top + slice_height
        
        # Extract the piece using OpenCV (y:y+h, x:x+w)
        slice_img = img[top:bottom, left:right]
        
        # Apply random rotation
        rotation_angle = random.choice(ROTATION_ANGLES)
        rotated_slice = rotate_image(slice_img, rotation_angle)
        
        # Generate filename for the piece (using random index)
        slice_filename = f"{base_name}_slice_{idx:03d}.png"
        slice_path = os.path.join(specific_output_dir, slice_filename)
        
        # Save the rotated piece
        cv2.imwrite(slice_path, rotated_slice)
        
        # Add order information for reconstruction
        slice_order.append({
            'filename': slice_filename,
            'row': row,
            'col': col,
            'original_position': original_position,
            'saved_as_index': idx,
            'rotation': rotation_angle,  # Store applied rotation
            'correct_rotation': (360 - rotation_angle) % 360,  # Rotation needed to restore
            'coordinates': {'left': left, 'top': top, 'right': right, 'bottom': bottom}
        })
    
    # Generate text file with correct order
    order_filename = f"{base_name}_order.txt"
    order_path = os.path.join(specific_output_dir, order_filename)
    
    with open(order_path, 'w', encoding='utf-8') as f:
        f.write(f"[VERSION 2 - WITH ROTATION]\n")
        f.write(f"Reconstruction information for: {os.path.basename(image_path)}\n")
        f.write(f"Original image: {img_width}x{img_height}\n")
        f.write(f"Division: {sqrt_slices}x{sqrt_slices} ({num_slices} pieces)\n")
        f.write(f"Piece size: {slice_width}x{slice_height}\n")
        f.write("NOTE: Pieces were saved in RANDOM ORDER with RANDOM ROTATION\n")
        f.write("-" * 70 + "\n\n")
        
        # Write detailed information for each piece (sorted by original position)
        f.write("\nCORRECT ORDER FOR RECONSTRUCTION:\n")
        f.write("Pos.Orig | File                | Row  | Col | Idx  | Applied Rot | Correct Rot\n")
        f.write("-" * 85 + "\n")
        
        # Sort by original position to show correct order
        sorted_slices = sorted(slice_order, key=lambda x: x['original_position'])
        
        for slice_info in sorted_slices:
            f.write(f"{slice_info['original_position']:8d} | {slice_info['filename']:18s} | "
                   f"{slice_info['row']:4d} | {slice_info['col']:3d} | "
                   f"{slice_info['saved_as_index']:4d} | "
                   f"{slice_info['rotation']:11d}° | "
                   f"{slice_info['correct_rotation']:10d}°\n")
        
        f.write("\n" + "-" * 70 + "\n")
        f.write("FILE MAPPING (File -> Original Position + Rotation):\n")
        # Sort by saved file index
        sorted_by_file = sorted(slice_order, key=lambda x: x['saved_as_index'])
        for slice_info in sorted_by_file:
            f.write(f"{slice_info['filename']} -> Position {slice_info['original_position']} "
                   f"(Row {slice_info['row']}, Col {slice_info['col']}) "
                   f"Rotate {slice_info['correct_rotation']}° to restore\n")
        
        f.write("\n" + "-" * 70 + "\n")
        f.write("ORIGINAL COORDINATES:\n")
        for slice_info in sorted_slices:
            coords = slice_info['coordinates']
            f.write(f"{slice_info['filename']}: "
                   f"({coords['left']}, {coords['top']}) -> "
                   f"({coords['right']}, {coords['bottom']}) "
                   f"[Rotated {slice_info['rotation']}°]\n")
    
    print(f"✓ Image successfully divided into {num_slices} parts WITH ROTATION")
    print(f"✓ Pieces saved to: {specific_output_dir}/")
    print(f"✓ Order file created: {order_path}")
    
    # Print rotation summary
    rotation_counts = {}
    for info in slice_order:
        rot = info['rotation']
        rotation_counts[rot] = rotation_counts.get(rot, 0) + 1
    print(f"✓ Rotation distribution: {rotation_counts}")
    
    return slice_order


def main():
    # Only execute interactive mode when running directly (not imported)
    parser = argparse.ArgumentParser(
        description='Divide an image into square pieces WITH RANDOM ROTATION (V2)')
    parser.add_argument('image_path', nargs='?', 
                       help='Path to the image to divide')
    parser.add_argument('-n', '--num_slices', type=int, default=9,
                       help='Number of pieces (default: 9, must have exact square root)')
    parser.add_argument('-o', '--output', default='sliced_images_v2',
                       help='Output folder (default: sliced_images_v2)')
    
    args = parser.parse_args()
    
    # Interactive mode if no image path provided
    if not args.image_path:
        print("\n=== IMAGE SLICER V2 (WITH ROTATION) ===")
        print("Available images in 'images/' folder:")
        
        images_dir = "images"
        if os.path.exists(images_dir):
            images = [f for f in os.listdir(images_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            for i, img in enumerate(images, 1):
                print(f"  {i}. {img}")
            
            if images:
                choice = input("\nEnter image number or full path: ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(images):
                        args.image_path = os.path.join(images_dir, images[idx])
                except ValueError:
                    args.image_path = choice
        
        if not args.image_path:
            args.image_path = input("Enter image path: ").strip()
        
        num_input = input(f"Number of pieces [{args.num_slices}]: ").strip()
        if num_input:
            args.num_slices = int(num_input)
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        return
    
    try:
        slice_image_v2(args.image_path, args.num_slices, args.output)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
