# Kintsugi 🧩

An intelligent image puzzle system that divides images into randomized pieces and then automatically reconstructs them using multiple AI algorithms. Named after the Japanese art of repairing broken pottery with gold, symbolizing the beauty found in reconstruction.

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
- **Configurable Border Width**: Adjust analysis width (10-100 pixels) for optimal matching
- **Organized Output**: Structured file organization by version, image name and slice count
- **Interactive & CLI Modes**: Both user-friendly interactive mode and command-line interface
- **Comprehensive Validation**: Input validation and error handling throughout the pipeline

## 🚀 Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install opencv-python numpy
```

### Basic Usage

1. **Interactive Mode** (Recommended for beginners):
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

# Results will be saved to:
# - sliced_images/example_16slices/ (puzzle pieces)
# - output_images/example_16slices/ (reconstructed images)
```

### Advanced: Configurable Border Width

You can adjust the border analysis width (default: 10 pixels) for fine-tuning algorithm performance:

```python
from puzzle_reconstructor.color_reconstructor import ColorSolver

# Default border width (10 pixels)
solver = ColorSolver("sliced_images/example_16slices", 
                     "output_images", 
                     "example")

# Custom border width (30 pixels for more context)
solver = ColorSolver("sliced_images/example_16slices", 
                     "output_images", 
                     "example",
                     border_width=30)

# Larger border for high-resolution puzzles (100 pixels)
solver = ColorSolver("sliced_images/example_16slices", 
                     "output_images", 
                     "example",
                     border_width=100)
```

**Border Width Guidelines:**
- **10-30px**: Recommended for standard puzzles (piece size 100-300px)
- **30-50px**: Better for noisy images or subtle gradients
- **50-100px**: Use for high-resolution pieces (>500px) or when edges are ambiguous
- **Avoid**: Border width > 50% of piece size (includes too much interior texture)

## 📁 Project Structure

```
Kintsugi/
├── main.py                     # Main orchestrator script
├── slice_images.py             # Image slicing functionality
├── puzzle_solver.py            # Unified reconstruction interface
├── puzzle_reconstructor/       # Reconstruction algorithms package
│   ├── __init__.py
│   ├── puzzle_base.py          # Base class for all solvers
│   ├── gradient_reconstructor.py   # Gradient-based reconstruction
│   ├── color_reconstructor.py      # Color-based reconstruction
│   └── random_reconstructor.py     # Random baseline reconstruction
├── images/                     # Input images directory
├── sliced_images/             # Generated puzzle pieces
│   └── [imagename]_[N]slices/ # Organized by image and slice count
└── output_images/             # Reconstructed results
    └── [imagename]_[N]slices/ # Results organized by method
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

### Gradient Analysis Algorithm

1. **Edge Detection**: Apply Sobel filters to detect horizontal/vertical edges
2. **Boundary Analysis**: Extract edge pixels along piece boundaries
3. **Compatibility Scoring**: Calculate gradient similarity between potential neighbors
4. **Optimization**: Use greedy placement with backtracking for optimal arrangement

### Color Analysis Algorithm

1. **Color Space Conversion**: Convert to LAB color space for perceptual accuracy
2. **Edge Sampling**: Extract color values along piece boundaries
3. **Distance Calculation**: Compute Euclidean distance in LAB space
4. **Matching**: Find best color matches between piece edges

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
pip install opencv-python numpy
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

## 🐛 Troubleshooting

### Common Issues

1. **"No perfect square" error**:
   - Solution: Use valid slice counts (4, 9, 16, 25, 36, etc.)

2. **Import errors**:
   - Solution: Ensure OpenCV and NumPy are installed: `pip install opencv-python numpy`

3. **Image not found**:
   - Solution: Check file path and supported formats

4. **Permission errors**:
   - Solution: Ensure write permissions for output directories

### Debugging Tips

- Run with verbose output to see detailed processing steps
- Check that input images are not corrupted
- Verify sufficient disk space for output files
- Ensure Python 3.6+ compatibility

### Development Setup

```bash
# Install development dependencies
pip install opencv-python numpy pytest

# Run tests (if available)
pytest tests/

# Format code
black *.py puzzle_reconstructor/*.py
```

## 🎯 Future Enhancements

- [ ] Advanced piece rotation handling
- [ ] Other methods for reconstruction:
    - [ ] MGC (Mahalanobis Gradient Covariance): "Jigsaw Puzzles with Pieces of Unknown Orientation" (Gallagher & Chen, CVPR 2012)
    - [ ] Minimum Spanning Tree (MST)

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
