# Kintsugi 🧩

An intelligent image puzzle system that divides images into randomized pieces and then automatically reconstructs them using multiple algorithms. Named after the Japanese art of repairing broken pottery with gold, symbolizing the beauty found in reconstruction.

## 🌟 Features

- **Two Puzzle Versions**:
  - **V1 (Standard)**: Pieces are shuffled but not rotated - easier to solve
  - **V2 (Advanced)**: Pieces are shuffled AND randomly rotated (0°, 90°, 180°, 270°) - more challenging
  - **Dual Execution**: Run both versions simultaneously to compare complexity
- **Image Slicing**: Automatically divide any image into square puzzle pieces with randomized arrangement
- **Multiple Reconstruction Algorithms**: 
  - **Paikin Solver (Best Buddies)**: Advanced algorithm using best-first search with multi-channel edge compatibility (LAB color + gradient)
  - **Gradient Analysis**: Uses edge detection and gradient matching for intelligent piece placement
  - **Color Analysis**: Employs LAB color space for perceptually accurate edge matching
  - **Random Assembly**: Baseline comparison using random piece arrangement
- **Advanced Reconstruction Engine**:
  - **Connection Graph**: Builds a complete weighted graph of all possible piece connections
  - **Greedy Selection**: Globally optimal edge selection using min-heap priority queue
  - **Union-Find**: Efficient cycle prevention and component tracking
  - **Backtracking**: Spatial layout optimization with BFS traversal
- **Benchmarking Suite**: Comprehensive performance evaluation across multiple puzzle sizes and datasets
- **Border Analysis Tools**: Visualize and compare edge compatibility between puzzle pieces
- **Clean Grid Generation**: Create visualization-ready puzzle reconstructions without metrics
- **Configurable Border Width**: Adjust analysis width (10-100 pixels) for optimal matching
- **Organized Output**: Structured file organization by version, image name and slice count
- **Interactive & CLI Modes**: Both user-friendly interactive mode and command-line interface
- **Comprehensive Validation**: Input validation and error handling throughout the pipeline

## 🚀 Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install -r requirements.txt

# Or manually:
pip install opencv-python numpy networkx
```

### Basic Usage

1. **Interactive Mode**:
```bash
python main.py
# Then select:
# 1 = V1 (standard puzzle, no rotation)
# 2 = V2 (advanced puzzle with rotation)
# 3 = Both versions (compare V1 vs V2)
```

2. **Command Line Mode**:
```bash
python main.py path/to/image.jpg 16 gradient
# You can also pass just the filename if the image is in the `images/` folder:
# python main.py example.jpg 16 gradient
```

3. **All reconstruction methods**:
```bash
python main.py path/to/image.jpg 16 all
```

4. **Execute both V1 and V2 versions**:
```bash
python main.py
# Select option 3 to run both versions and compare results
```

### Example

```bash
# Slice an image into 16 pieces and reconstruct using all methods
python main.py images/example.jpg 16 all
# Or simply:
# python main.py example.jpg 16 all   # resolves to images/example.jpg if present

# Results will be saved to:
# - sliced_images/example_16slices/ (puzzle pieces)
# - output_images/example_16slices/ (reconstructed images)
```

### Advanced: Configurable Border Width

You can adjust the border analysis width (default: 100 pixels) for fine-tuning algorithm performance:

```python
from puzzle_reconstructor.color_reconstructor import ColorSolver

# Default border width (100 pixels)
solver = ColorSolver("sliced_images/example_16slices", 
                     "output_images", 
                     "example")

# Custom border width (120 pixels for more context)
solver = ColorSolver("sliced_images/example_16slices", 
                     "output_images", 
                     "example",
                     border_width=120)

# Larger border for high-resolution puzzles (200 pixels)
solver = ColorSolver("sliced_images/example_16slices", 
                     "output_images", 
                     "example",
                     border_width=200)
```

**Border Width Guidelines:**
- **10-30px**: Recommended for standard puzzles (piece size 100-300px)
- **30-50px**: Better for noisy images or subtle gradients
- **50-100px**: Use for high-resolution pieces (>500px) or when edges are ambiguous
- **Avoid**: Border width > 50% of piece size (includes too much interior texture)

## 📁 Project Structure

```
Kintsugi/
├── main.py                                    # Main orchestrator script
├── slice_images.py                            # Image slicing V1 (no rotation)
├── slice_images_v2.py                         # Image slicing V2 (with rotation)
├── puzzle_solver.py                           # Unified reconstruction interface
├── benchmark_puzzle_v1_v2_multi_grids.py      # Comprehensive benchmarking tool
├── analyze_border_comparison_color.py         # Color-based border analysis
├── analyze_border_comparison_gradient.py      # Gradient-based border analysis
├── analyze_border_comparison_paikin.py        # Paikin-based border analysis
├── analyze_border_slice_6_vs_7.py             # Border analysis for specific slices
├── generate_clean_grid.py                     # Clean grid generation V1
├── generate_clean_grid_v2.py                  # Clean grid generation V2
├── extract_borders.py                         # Border extraction utility
├── requirements.txt                           # Project dependencies
├── puzzle_reconstructor/                      # V1 Reconstruction algorithms
│   ├── __init__.py
│   ├── puzzle_base.py                         # Base class with connection graph + backtracking
│   ├── gradient_reconstructor.py              # Gradient-based reconstruction
│   ├── color_reconstructor.py                 # Color-based reconstruction
│   ├── paikin_reconstructor.py                # Paikin Best Buddies algorithm
│   └── random_reconstructor.py                # Random baseline reconstruction
├── puzzle_reconstructor_v2/                   # V2 Reconstruction (with rotation)
│   ├── __init__.py
│   ├── puzzle_base_v2.py                      # Base class with rotation support
│   ├── gradient_reconstructor_v2.py           # Gradient with rotation support
│   ├── color_reconstructor_v2.py              # Color with rotation support
│   ├── paikin_reconstructor_v2.py             # Paikin with rotation support
│   └── random_reconstructor_v2.py             # Random with rotation support
├── images/                                    # Input images directory
├── datasets/                                  # Dataset directory (e.g., CIFAR-10)
├── sliced_images_v1/                          # V1 puzzle pieces (no rotation)
│   └── [imagename]_[N]slices/
│       ├── [imagename]_slice_000.png
│       ├── [imagename]_slice_001.png
│       ├── ...
│       └── [imagename]_order.txt              # Solution mapping
├── sliced_images_v2/                          # V2 puzzle pieces (with rotation)
│   └── [imagename]_[N]slices/
│       ├── [imagename]_slice_000.png          # Fixed anchor (top-left, no rotation)
│       ├── [imagename]_slice_001.png
│       ├── ...
│       └── [imagename]_order.txt              # Solution mapping with rotations
├── output_images/                             # Reconstructed results
│   ├── ver_1/                                 # V1 reconstructions
│   │   └── [imagename]_[N]slices/
│   │       ├── paikin_reconstructed.png
│   │       ├── paikin_reconstruction_map.txt
│   │       ├── gradient_reconstructed.png
│   │       ├── gradient_reconstruction_map.txt
│   │       ├── color_reconstructed.png
│   │       ├── color_reconstruction_map.txt
│   │       ├── random_reconstructed.png
│   │       └── random_reconstruction_map.txt
│   └── ver_2/                                 # V2 reconstructions
│       └── [imagename]_[N]slices/
│           ├── paikin_v2_reconstructed.png
│           ├── paikin_v2_reconstruction_map.txt
│           ├── gradient_v2_reconstructed.png
│           ├── gradient_v2_reconstruction_map.txt
│           ├── color_v2_reconstructed.png
│           ├── color_v2_reconstruction_map.txt
│           ├── random_v2_reconstructed.png
│           └── random_v2_reconstruction_map.txt
├── benchmark_output/                          # Benchmark results
│   ├── results.json                           # Detailed results
│   └── benchmark_summary.csv                  # CSV summary
├── border_analysis_output/                    # Border comparison visualizations
└── borders_[imagename]_slice_[N]/             # Extracted borders for analysis
```

## 🛠️ Components

### Image Slicer (`slice_images.py`)

Divides input images into square puzzle pieces with the following features:

- **Smart Grid Division**: Automatically calculates optimal piece dimensions
- **Random Shuffling**: Pieces are randomly arranged to create a proper puzzle
- **Metadata Generation**: Creates solution files for validation
- **Format Support**: JPEG, PNG, BMP, TIFF, WebP

**Usage:**
```bash
python slice_images.py image.jpg 16
# You can also pass a filename that exists in the `images/` folder:
# python slice_images.py example.jpg 16
```

### Reconstruction Algorithms

#### 1. Gradient Reconstructor 🔍
- Uses Sobel edge detection to analyze piece boundaries
- Calculates gradient compatibility between adjacent pieces
- Optimizes for smooth edge transitions

#### 2. Color Reconstructor 🎨
- Analyzes edge colors in perceptually uniform LAB color space
- Minimizes color differences at piece boundaries
- Accounts for human color perception

#### 3. Random Reconstructor 🎲
- Provides baseline comparison
- Useful for algorithm validation
- Demonstrates improvement over chance

### Unified Interface (`puzzle_solver.py`)

Interactive command-line tool for running reconstruction algorithms:

```bash
python puzzle_solver.py
```

Features:
- Image discovery and selection
- Algorithm comparison
- Progress tracking
- Result validation

## 📊 Algorithm Details

### Connection Graph Reconstruction (Base Algorithm)

All reconstruction methods use a unified connection graph approach:

1. **Complete Graph Construction**: 
   - Build weighted graph of ALL possible piece connections (O(n²) edges)
   - Each edge represents compatibility between two piece borders
   - Supports 4 connection types: right, bottom, left, top

2. **Global Greedy Selection**:
   - Use min-heap to process connections by descending compatibility score
   - Select globally best connections (not just locally optimal)
   - Union-Find data structure prevents cycles and tracks components

3. **Edge Constraint Enforcement**:
   - Maximum 4 edges per piece (one per side)
   - No duplicate edges on the same piece side
   - Connected graph formation ensures single component

4. **Spatial Layout via BFS**:
   - Convert connection graph to 2D spatial grid
   - Breadth-first traversal from anchor piece
   - Dynamic grid expansion to accommodate all pieces

5. **Backtracking & Optimization**:
   - If layout conflicts arise, backtrack to last valid state
   - Reorder placement queue based on connectivity strength
   - Ensures all pieces fit in rectangular grid

### Gradient Analysis Algorithm

1. **Edge Detection**: Apply Sobel (3x3) filters to detect horizontal/vertical edges
2. **Boundary Analysis**: Extract configurable-width borders (10-100px) along piece edges
3. **Compatibility Scoring**: Calculate gradient continuity across junction using LAB color space
4. **Quality Index**: Combine gradient differences (40%), color differences (40%), and smoothness metrics (20%)
5. **Graph Integration**: Feed compatibility scores into connection graph for reconstruction

### Color Analysis Algorithm

1. **Color Space Conversion**: Convert to LAB color space for perceptual accuracy
2. **Edge Sampling**: Extract color values along configurable-width borders
3. **Distance Calculation**: Compute Euclidean distance in LAB space between adjacent edges
4. **Compatibility Scoring**: Lower distance = higher compatibility
5. **Graph Integration**: Use color distances as edge weights in connection graph

### Paikin Best Buddies Algorithm

1. **Multi-Channel Features**: Combine LAB color (60%) and gradient magnitude (40%)
2. **Best Buddies Criterion**: Pieces A and B are "best buddies" if each is the other's best match
3. **Best-First Search**: Priority queue expansion starting from fixed anchor piece
4. **Rotation Support (V2)**: Tests all 4 rotations (0°, 90°, 180°, 270°) per piece
5. **Greedy Placement**: Places pieces one-by-one based on lowest compatibility cost to existing layout

## 📈 Performance

The system supports various puzzle sizes:
- **4 pieces** (2×2): Nearly instant reconstruction
- **16 pieces** (4×4): Fast reconstruction (~1-5 seconds)
- **64 pieces** (8×8): Moderate complexity (~10-30 seconds)
- **256+ pieces**: Advanced puzzles (timing varies)

**Note**: Valid slice counts must be perfect squares (4, 9, 16, 25, 36, 49, 64, 81, 100, etc.)

## 🔧 Configuration

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

### Environment Setup

1. **Clone the repository**:
```bash
git clone https://github.com/candeladavila/Kintsugi.git
cd Kintsugi
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt

# Or manually:
pip install opencv-python numpy networkx
```

## 📝 Usage Examples

### Example 1: Quick Test
```bash
# Place an image in the images/ folder
python main.py images/test.jpg 9 gradient
```

### Example 2: Compare All Methods
```bash
python main.py images/landscape.jpg 25 all
```

### Example 3: Batch Processing
```bash
# Process multiple images
for img in images/*.jpg; do
    python main.py "$img" 16 all
done
```

## � Advanced Tools

### Benchmarking Suite

Evaluate reconstruction algorithms across multiple images and puzzle sizes:

```bash
python benchmark_puzzle_v1_v2_multi_grids.py \
    --num-images 50 \
    --grid-sizes 3,5,10 \
    --timeout 300 \
    --border-width 10
```

Features:
- Tests on CIFAR-10 dataset (automatic download)
- Multiple grid sizes (3×3, 5×5, 10×10, etc.)
- Timeout protection to prevent infinite loops
- Generates JSON and CSV summaries
- Calculates piece accuracy, border accuracy, and runtime metrics

### Border Analysis Tools

#### Compare Specific Slice Borders
```bash
python compare_slice_borders.py
```
Generates visual comparison of border matching between two specific puzzle pieces.

#### Analyze All Border Combinations
```bash
python analyze_border_slice_6_vs_7.py
```
Creates 4 visualizations comparing one slice's border against all 4 borders of another slice.

### Clean Grid Generation

Generate visualization-ready puzzle reconstructions without metrics overlay:

```bash
# For V1 reconstructions
python generate_clean_grid.py

# For V2 reconstructions (with rotation)
python generate_clean_grid_v2.py
```

Outputs clean grid images with only border lines, perfect for presentations and papers.

## 🐛 Troubleshooting

### Common Issues

1. **"No perfect square" error**:
   - Solution: Use valid slice counts (4, 9, 16, 25, 36, etc.)

2. **Import errors**:
   - Solution: Ensure OpenCV and NumPy are installed: `pip install opencv-python numpy`

3. **Timeout errors in benchmark**:
   - Solution: Increase `--timeout` parameter or reduce grid size

4. **Image not found**:
   - Solution: Check file path and supported formats

5. **Permission errors**:
   - Solution: Ensure write permissions for output directories

### Debugging Tips

- Run with verbose output to see detailed processing steps
- Check that input images are not corrupted
- Verify sufficient disk space for output files
- Ensure Python 3.6+ compatibility
- For benchmarks, start with small `--num-images` to test

### Development Setup

```bash
# Install development dependencies
pip install opencv-python numpy matplotlib torch torchvision

# For benchmarking (CIFAR-10 dataset)
pip install torch torchvision pillow

# Run tests (if available)
pytest tests/

# Format code
black *.py puzzle_reconstructor/*.py puzzle_reconstructor_v2/*.py
```

## 📊 Benchmark Results

The benchmark suite evaluates algorithms across multiple metrics:

- **Piece Accuracy Index**: Proportion of pieces in correct positions (0-1)
- **Border Accuracy Index**: Proportion of correct neighbor pairs (0-1)
- **Rotation Accuracy Index** (V2 only): Proportion of pieces with correct rotation (0-1)
- **Runtime**: Average execution time in seconds

Results are saved as:
- `benchmark_output/benchmark_summary.json` (detailed results)
- `benchmark_output/benchmark_summary.csv` (spreadsheet-friendly)

## 🎯 Future Enhancements

- [x] Advanced piece rotation handling (V2 implemented)
- [x] Comprehensive benchmarking suite
- [x] Border analysis visualization tools
- [ ] Other methods for reconstruction:
    - [ ] MGC (Mahalanobis Gradient Covariance): "Jigsaw Puzzles with Pieces of Unknown Orientation" (Gallagher & Chen, CVPR 2012)
    - [ ] Minimum Spanning Tree (MST)
- [ ] GPU acceleration for large puzzles
- [ ] Real-time reconstruction progress visualization

## 🙏 Acknowledgments

- Inspired by traditional jigsaw puzzles and the Japanese art of Kintsugi
- Built with OpenCV for robust image processing
- Uses NumPy for efficient numerical computations

## 👩🏻‍💻 Author
Made with ❤️ **Candela Dávila Moreno**

## ⚙️ Reporting Issues
If you encounter any bugs or issues, please feel free to contact me at [candeladavila05@gmail.com](mailto:candeladavila05@gmail.com)

---

*"In the Japanese art of Kintsugi, broken pottery is repaired with gold, making the repaired piece more beautiful than the original. Similarly, this project finds beauty in the reconstruction of fragmented images."* ✨
