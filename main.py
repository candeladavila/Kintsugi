#!/usr/bin/env python3
"""
KINTSUGI - Complete system for image puzzle slicing and reconstruction

VERSION 1 (V1): Standard puzzle - pieces are shuffled but not rotated
VERSION 2 (V2): Advanced puzzle - pieces are shuffled AND randomly rotated (0°, 90°, 180°, 270°)

Combines the functionality of slice_images.py and puzzle_solver.py in an automatic workflow
"""

import os
import sys
import math
from pathlib import Path


# =============================================================================
#   VERSION SELECTION
# =============================================================================

def select_version():
    """
    Asks the user to select which version of the system to use.
    
    Returns:
        int: 1 for V1 (no rotation), 2 for V2 (with rotation), 3 for both
    """
    print("\n" + "=" * 60)
    print("SELECT PUZZLE VERSION")
    print("=" * 60)
    print("")
    print("  VERSION 1 (V1): Standard Puzzle")
    print("    - Pieces are randomly shuffled")
    print("    - No rotation applied")
    print("")
    print("  VERSION 2 (V2): Advanced Puzzle with Rotation")
    print("    - Pieces are randomly shuffled")
    print("    - Each piece rotated 0°, 90°, 180° or 270°")
    print("")
    print("  BOTH VERSIONS: Execute V1 and V2 simultaneously")
    print("    - Generates two puzzle sets for comparison")
    print("    - Separate results in ver_1/ and ver_2/")
    print("")
    
    while True:
        choice = input("Select version (1, 2 or 3 for both): ").strip()
        if choice == '1':
            print("Selected: VERSION 1 (Standard)")
            return 1
        elif choice == '2':
            print("Selected: VERSION 2 (With Rotation)")
            return 2
        elif choice == '3':
            print("Selected: BOTH VERSIONS (V1 + V2)")
            return 3
        else:
            print("Please enter 1, 2 or 3")

# =============================================================================
#   V1 FUNCTIONS (No Rotation) - Original
# =============================================================================

def run_slice_images_v1(image_path, num_slices):
    """
    Executes slice_images.py with the specified parameters (V1 - no rotation)
    """
    print("Starting image slicing")
    
    try:
        # Import and execute directly instead of subprocess
        import slice_images
        slice_images.slice_image(image_path, num_slices, "sliced_images_v1")
        return True
        
    except Exception as e:
        print(f"Error during slicing: {e}")
        return False

def run_puzzle_solver_v1(image_name, num_slices, method='all', border_width=100):
    """
    Executes puzzle_solver.py with the specified configuration (V1 - no rotation)
    """
    
    try:
        # Find the specific folder created by slice_images
        sliced_dir = f"sliced_images_v1/{image_name}_{num_slices}slices"
        if not os.path.exists(sliced_dir):
            print(f"Error: Folder not found {sliced_dir}")
            return False
        
        # Import required modules
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'puzzle_reconstructor'))
        from gradient_reconstructor import GradientSolver
        from color_reconstructor import ColorSolver
        from random_reconstructor import RandomSolver
        from paikin_reconstructor import PaikinSolver
        
        output_dir = "output_images/ver_1"
        
        # Execute methods according to selection
        if method == 'all':
            methods_to_run = [
                ('paikin', PaikinSolver),
                ('gradient', GradientSolver),
                ('color', ColorSolver),
                ('random', RandomSolver)
            ]
        else:
            solver_map = {
                'paikin': PaikinSolver,
                'gradient': GradientSolver,
                'color': ColorSolver,
                'random': RandomSolver
            }
            if method not in solver_map:
                print(f"Error: Unknown method '{method}'")
                return False
            methods_to_run = [(method, solver_map[method])]
        
        success_count = 0
        for method_name, solver_class in methods_to_run:
            try:
                print(f"\nExecuting method: {method_name.upper()}")
                print(f"Algorithm: Global Affinity Graph + Backtracking")
                print(f"Border width: {border_width}px")
                solver = solver_class(sliced_dir, output_dir, image_name, border_width)
                solver.load_slices(image_name)
                solver.solve()
                success_count += 1
            except Exception as e:
                print(f"Error in method {method_name}: {e}")
        
        if success_count > 0:
            print(f"\nReconstruction completed")
            return True
        else:
            print("All methods failed")
            return False
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Verify that puzzle_reconstructor modules are available")
        return False
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        return False

# =============================================================================
#   V2 FUNCTIONS (With Rotation)
# =============================================================================

def run_slice_images_v2(image_path, num_slices):
    """
    Slices image using V2 method (with random rotation).
    """
    
    try:
        import slice_images_v2
        slice_images_v2.slice_image_v2(image_path, num_slices, "sliced_images_v2")
        return True
    except Exception as e:
        print(f"Error during slicing: {e}")
        return False

def run_puzzle_solver_v2(image_name, num_slices, method='all', border_width=100):
    """
    Reconstructs puzzle using V2 solvers (with rotation support).
    """
    
    try:
        sliced_dir = f"sliced_images_v2/{image_name}_{num_slices}slices"
        if not os.path.exists(sliced_dir):
            print(f"Error: Folder not found {sliced_dir}")
            return False
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'puzzle_reconstructor_v2'))
        from gradient_reconstructor_v2 import GradientSolverV2
        from color_reconstructor_v2 import ColorSolverV2
        from random_reconstructor_v2 import RandomSolverV2
        from paikin_reconstructor_v2 import PaikinSolverV2
        
        output_dir = "output_images/ver_2"
        
        if method == 'all':
            methods_to_run = [
                ('paikin', PaikinSolverV2),
                ('gradient', GradientSolverV2),
                ('color', ColorSolverV2),
                ('random', RandomSolverV2)
            ]
        else:
            solver_map = {
                'paikin': PaikinSolverV2,
                'gradient': GradientSolverV2,
                'color': ColorSolverV2,
                'random': RandomSolverV2
            }
            if method not in solver_map:
                print(f"Error: Unknown method '{method}'")
                return False
            methods_to_run = [(method, solver_map[method])]
        
        success_count = 0
        for method_name, solver_class in methods_to_run:
            try:
                print(f"\nExecuting method: {method_name.upper()} V2")
                print(f"Algorithm: Rotation-Aware Global Graph + Recursive Backtracking")
                print(f"Border width: {border_width}px")
                solver = solver_class(sliced_dir, output_dir, image_name, border_width)
                solver.load_slices(image_name)
                solver.solve()
                success_count += 1
            except Exception as e:
                print(f"Error in method {method_name}: {e}")
                import traceback
                traceback.print_exc()
        
        if success_count > 0:
            return True
        else:
            print("All methods failed")
            return False
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Verify that puzzle_reconstructor_v2 is available")
        return False
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        return False

# =============================================================================
#   UTILITY FUNCTIONS
# =============================================================================

def get_image_name(image_path):
    """Extracts the base name of the image without extension"""
    return Path(image_path).stem

def validate_image_exists(image_path):
    """Validate that the image exists either as given or inside `images/`.

    Returns the resolved path if found, otherwise `None`.
    """
    # Direct path
    if os.path.exists(image_path):
        resolved = image_path
    else:
        # Try inside images/ directory
        candidate = os.path.join("images", image_path)
        if os.path.exists(candidate):
            resolved = candidate
        else:
            # Try common extensions if user provided base name without extension
            name, ext = os.path.splitext(image_path)
            if ext == "":
                for ext_try in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                    candidate_ext = os.path.join("images", name + ext_try)
                    if os.path.exists(candidate_ext):
                        resolved = candidate_ext
                        break
                else:
                    resolved = None
            else:
                resolved = None

    if not resolved:
        print(f"Error: File not found {image_path}")
        return None

    # Verify that it's a valid image file by extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    ext = Path(resolved).suffix.lower()
    if ext not in valid_extensions:
        print(f"Error: {ext} is not a supported image format")
        print(f"Valid formats: {', '.join(valid_extensions)}")
        return None

    return resolved

def validate_num_slices(num_slices):
    """Validates that the number of slices has an exact square root"""
    import math
    sqrt_slices = int(math.sqrt(num_slices))
    if sqrt_slices * sqrt_slices != num_slices:
        print(f"Error: {num_slices} does not have an exact square root")
        print(f"Valid numbers: 4, 9, 16, 25, 36, 49, 64, 81, 100, etc.")
        return False
    return True

def interactive_mode(version=1):
    """Interactive mode to get parameters from user"""
    if version == 3:
        version_str = "BOTH (V1 + V2)"
    else:
        version_str = "V1 (Standard)" if version == 1 else "V2 (With Rotation)"
    print(f"KINTSUGI - IMAGE PUZZLE SYSTEM - {version_str}")
    print("=" * 60)
    print("Graph-based reconstruction with backtracking and MST guidance")
    print("")
    
    # Get image path (without listing)
    while True:
        image_input = input("Image path: ").strip()
        if not image_input:
            print("Please enter a valid path")
            continue
        
        resolved = validate_image_exists(image_input)
        if resolved:
            image_path = resolved
            break
    
    # Get number of slices
    while True:
        try:
            num_input = input("Number of pieces (4, 9, 16, 25, etc.): ").strip()
            num_slices = int(num_input)
            
            if validate_num_slices(num_slices):
                break
        except ValueError:
            print("Please enter a valid square integer")
    
    # Select reconstruction method
    methods = {
        '1': ('paikin', 'Paikin Solver (Global Graph Analysis)'),
        '2': ('gradient', 'Gradient Solver (Structural Borders)'),
        '3': ('color', 'Color Solver (Chromatographic Borders)'),
        '4': ('random', 'Random Placement (Baseline)'),
        '5': ('all', 'All Methods')
    }
    
    print("\nReconstruction methods:")
    for key, (method, desc) in methods.items():
        print(f"  {key}. {method.upper()} - {desc}")
    
    while True:
        choice = input("\nSelect method (1-5, Enter for all): ").strip()
        
        if not choice:  # Enter pressed
            method = 'all'
            break
        elif choice in methods:
            method = methods[choice][0]
            break
        else:
            print("Please select a valid option (1-5)")
    
    # Configure border_width
    print("\nBorder width configuration:")
    print("  Border width determines how many pixels from edges are analyzed")
    print("  Default: 100 pixels (recommended for high-resolution images)")
    
    while True:
        border_input = input("\nUse default (100px)? (Enter for yes, or type a value): ").strip()
        
        if not border_input:  # Use default
            border_width = 100
            print("Using default: 100 pixels")
            break
        
        try:
            border_width = int(border_input)
            
            if border_width <= 0:
                print("Error: Border width must be positive")
                continue
            
            if border_width > 500:
                print("Warning: Very large border width (>500px) may include too much interior")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            
            print(f"Using border width: {border_width} pixels")
            break
            
        except ValueError:
            print("Error: Please enter a valid number")
    
    return image_path, num_slices, method, border_width

def main():
    """Main function"""
    print()
    print("=" * 60)
    print("KINTSUGI - Image Puzzle System")
    print("=" * 60)
    
    # Step 0: Select version FIRST
    version = select_version()
    
    # Process command line arguments
    if len(sys.argv) >= 3:
        # Command line mode
        image_path = sys.argv[1]
        try:
            num_slices = int(sys.argv[2])
        except ValueError:
            print("Error: Number of slices must be an integer")
            return
        
        method = sys.argv[3] if len(sys.argv) > 3 else 'all'
        border_width = 100  # Default for command line mode
        
        # Validations
        resolved = validate_image_exists(image_path)
        if not resolved:
            return
        image_path = resolved
        if not validate_num_slices(num_slices):
            return
        
    else:
        # Interactive mode - pass version for display
        image_path, num_slices, method, border_width = interactive_mode(version)
    
    # Get base image name
    image_name = get_image_name(image_path)
    
    if version == 3:
        version_str = "BOTH (V1 + V2)"
        output_folder = "output_images/ver_1 and output_images/ver_2"
    else:
        version_str = "V1 (Standard)" if version == 1 else "V2 (With Rotation)"
        output_folder = "output_images/ver_1" if version == 1 else "output_images/ver_2"
    
    print(f"\nConfiguration:")
    print(f"   Version: {version_str}")
    print(f"   Image: {image_path}")
    print(f"   Base name: {image_name}")
    print(f"   Pieces: {num_slices}")
    print(f"   Method: {method.upper()}")
    print(f"   Backtracking: Enabled")
    print(f"   MST Guidance: Enabled")
    print("")
    
    # Confirm before proceeding
    if len(sys.argv) < 3:  # Only in interactive mode
        confirm = input("Continue? (Enter for yes, 'n' for no): ").strip().lower()
        if confirm == 'n':
            print("Operation cancelled")
            return
    
    # Execute according to version
    if version == 1:
        # V1: Standard (no rotation)
        success = run_slice_images_v1(image_path, num_slices)
        if not success:
            print("\nImage slicing failed. Process terminated.")
            return
        
        success = run_puzzle_solver_v1(image_name, num_slices, method, border_width)
        if not success:
            print("\nPuzzle reconstruction failed.")
            return
            
    elif version == 2:
        # V2: With rotation
        success = run_slice_images_v2(image_path, num_slices)
        if not success:
            print("\nImage slicing failed. Process terminated.")
            return
        
        success = run_puzzle_solver_v2(image_name, num_slices, method, border_width)
        if not success:
            print("\nPuzzle reconstruction failed.")
            return
            
    else:  # version == 3: Both versions
        print("\n" + "=" * 60)
        print("PART 1/2: EXECUTING VERSION 1 (Standard)")
        print("=" * 60)
        
        success_v1 = run_slice_images_v1(image_path, num_slices)
        if success_v1:
            success_v1 = run_puzzle_solver_v1(image_name, num_slices, method, border_width)
        
        print("\n" + "=" * 60)
        print("PART 2/2: EXECUTING VERSION 2 (With Rotation)")
        print("=" * 60)
        
        success_v2 = run_slice_images_v2(image_path, num_slices)
        if success_v2:
            success_v2 = run_puzzle_solver_v2(image_name, num_slices, method, border_width)
        
        # Summary for both versions
        print("\n" + "=" * 60)
        if success_v1 and success_v2:
            print("BOTH VERSIONS COMPLETED SUCCESSFULLY")
        elif success_v1:
            print("COMPLETED WITH WARNINGS - Only V1 successful")
        elif success_v2:
            print("COMPLETED WITH WARNINGS - Only V2 successful")
        else:
            print("BOTH VERSIONS FAILED")
            return
        print("=" * 60)
        print(f"V1 pieces saved in: sliced_images_v1/{image_name}_{num_slices}slices/")
        print(f"V2 pieces saved in: sliced_images_v2/{image_name}_{num_slices}slices/")
        print(f"V1 results: output_images/ver_1/{image_name}_{num_slices}slices/")
        print(f"V2 results: output_images/ver_2/{image_name}_{num_slices}slices/")
        return
    
    # Final summary (for single version execution)
    print("\n" + "=" * 60)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    if version == 1:
        print(f"Pieces saved in: sliced_images_v1/{image_name}_{num_slices}slices/")
    else:
        print(f"Pieces saved in: sliced_images_v2/{image_name}_{num_slices}slices/")
    print(f"Results in: {output_folder}/{image_name}_{num_slices}slices/")
    if version == 2:
        print("Note: V2 includes rotation and relative connection accuracy metrics")
    print("\nCheck the results and analyze the graph-based reconstruction performance!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)