# DeepProstate

<p align="center">
  <img src="src/image/logo2.svg" alt="DeepProstate Logo" width="200"/>
</p>

<p align="center">
  <strong>Advanced AI-Powered Prostate MRI Analysis Platform</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/PyQt-6-green.svg" alt="PyQt6"/>
  <img src="https://img.shields.io/badge/AI-nnUNet-orange.svg" alt="nnUNet"/>
  <img src="https://img.shields.io/badge/Medical-DICOM-red.svg" alt="DICOM"/>
  <img src="https://img.shields.io/badge/License-Medical-lightgrey.svg" alt="License"/>
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [AI Models](#ai-models)
- [Supported Formats](#supported-formats)
- [User Guide](#user-guide)
- [Development](#development)
- [Quality Assurance](#quality-assurance)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## 🎯 Overview

**DeepProstate** is a professional medical imaging workstation designed for advanced prostate MRI analysis using state-of-the-art artificial intelligence. Built following **Clean Architecture** principles, it provides radiologists and researchers with powerful tools for automatic segmentation, quantitative analysis, and clinical decision support.

### Mission

To provide clinicians with accurate, reliable, and efficient AI-powered tools for prostate cancer detection and analysis, while maintaining the highest standards of medical software quality and regulatory compliance.

### Target Users

- **Radiologists**: Clinical interpretation and diagnosis
- **Urologists**: Treatment planning and follow-up
- **Researchers**: Medical imaging research and AI model validation
- **Medical Physicists**: Image quality assessment and protocol optimization

---

## ✨ Key Features

### 🤖 AI-Powered Analysis

- **Automatic Segmentation** using nnUNet v2 architecture
  - Prostate gland delineation
  - Transition Zone (TZ) and Peripheral Zone (PZ) segmentation
  - Clinically Significant Prostate Cancer (csPCa) detection
- **Multi-Sequence Support**: T2W, ADC, High B-Value (HBV)
- **Confidence Scoring** for quality assurance
- **Real-time Analysis** with progress tracking

### 🖼️ Advanced Visualization

- **Multi-Planar Reconstruction** (Axial, Sagittal, Coronal)
- **3D Volume Rendering** using VTK
- **Overlay Management** with adjustable opacity
- **Window/Level Presets** for different tissue types
- **Cross-hair Synchronization** across views
- **Measurement Tools** (distance, area, volume)

### 📊 Quantitative Analysis

- **Radiomics Features**: texture, shape, intensity metrics
- **Volume Calculations** with spatial calibration
- **Statistical Analysis**: mean, median, standard deviation
- **Histogram Analysis** for intensity distribution
- **Export to CSV/Excel** for further analysis

### ✏️ Manual Editing

- **Brush Tools** for segmentation refinement
- **Multi-Label Support** for complex anatomical structures
- **Undo/Redo** functionality
- **Mask Merging** and splitting
- **Smart Interpolation** between slices

### 🔄 Format Support

- **DICOM** (Digital Imaging and Communications in Medicine)
- **NIfTI** (Neuroimaging Informatics Technology Initiative)
- **MHA/MHD** (MetaImage format)
- **NRRD** (Nearly Raw Raster Data)
- Automatic format detection and conversion

### 🛡️ Medical Compliance

- **HIPAA-Compliant** logging and data handling
- **Medical Audit Trail** with timestamped actions
- **Patient Privacy Protection** with data anonymization
- **Validation Reports** for regulatory compliance
- **Secure Configuration** management

---

## 🏗️ Architecture

DeepProstate follows **Clean Architecture** principles with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                  Presentation Layer                     │
│         (PyQt6 UI, Widgets, Visualization)              │
├─────────────────────────────────────────────────────────┤
│              Application Services Layer                 │
│    (Use Cases, Orchestrators, Business Logic)           │
├─────────────────────────────────────────────────────────┤
│                  Domain Layer                           │
│   (Entities, Value Objects, Domain Services)            │
├─────────────────────────────────────────────────────────┤
│              Infrastructure Layer                       │
│  (Repositories, External Services, Frameworks)          │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **UI Framework** | PyQt6 |
| **AI Engine** | nnUNet v2 |
| **3D Rendering** | VTK (Visualization Toolkit) |
| **Medical Imaging** | pydicom, nibabel, SimpleITK |
| **Numerical Computing** | NumPy, SciPy |
| **Image Processing** | scikit-image |
| **Dependency Injection** | Custom Medical Service Container |

---

## 💻 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Disk Space**: 10GB+ free space
- **OS**: Linux, Windows, macOS
- **GPU**: Optional (CUDA-compatible for faster inference)

### Step 1: Clone Repository

```bash
git clone https://github.com/Marquita-oss/DeepProstate.git
cd deep-prostate
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python3 -m venv medical-env
source medical-env/bin/activate  # Linux/macOS
# medical-env\Scripts\activate  # Windows

# Or using conda
conda create -n deep-prostate python=3.8
conda activate deep-prostate
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install optional dependencies for full functionality
pip install nibabel SimpleITK vtk pydicom scikit-image scipy

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Verify Installation

```bash
python -c "import PyQt6, numpy, pydicom; print('✓ Core dependencies OK')"
```

---

### First Time Setup

1. **Load AI Models**
   - Click "📁 Load AI Models Path" in AI Analysis panel
   - Select directory containing nnUNet models
   - Wait for model validation (~30 seconds)

2. **Load Patient Data**
   - Use "Patient Browser" panel
   - Click "Load DICOM Folder" or "Load Single File"
   - Supported formats: DICOM, NIfTI, MHA, NRRD

3. **Run AI Analysis**
   - Select loaded image in Patient Browser
   - Go to "AI Analysis" panel
   - Choose analysis type (Prostate Gland, TZ/PZ Zones, csPCa Detection)
   - Click "Run AI Analysis"
   - Review results in 2D/3D viewers

### Example Workflow

```python
# 1. Load patient MRI
Patient Browser → Load DICOM Folder → Select T2W_AXIAL

# 2. Run automatic segmentation
AI Analysis → Select "Prostate Gland" → Run AI Analysis

# 3. Review results
View segmentation overlay in Axial/Sagittal/Coronal views
Adjust opacity slider for better visualization

# 4. Manual refinement (optional)
Manual Editing → Select Brush Tool → Refine boundaries

# 5. Quantitative analysis
Quantitative Analysis → View volume, intensity statistics
Export results to CSV

# 6. 3D visualization
Toggle 3D view → Rotate/zoom prostate model
```

---

## 🧠 AI Models

### nnUNet v2 Integration

DeepProstate uses **nnUNet** (no-new-Net), a self-configuring deep learning framework for medical image segmentation.

#### Supported Analysis Types

1. **Prostate Gland Segmentation**
   - **Input**: T2-weighted MRI
   - **Output**: Complete prostate gland mask
   - **Use Case**: Volume calculation, treatment planning

2. **Zonal Anatomy (TZ/PZ)**
   - **Input**: T2-weighted MRI
   - **Output**: Transition Zone and Peripheral Zone masks
   - **Use Case**: PI-RADS assessment, focal therapy planning

3. **csPCa Detection**
   - **Input**: Multi-sequence (T2W + ADC + HBV)
   - **Output**: Clinically significant cancer lesion masks
   - **Use Case**: Cancer detection, biopsy targeting

### Model Requirements

```
models/
├── Task998_ProstateGland/
│   └── nnUNetTrainer__nnUNetPlans__3d_fullres/
├── Task600_ProstateTZPZ/
│   └── nnUNetTrainer__nnUNetPlans__3d_fullres/
└── Task500_csPCa/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/
```

### Performance Metrics

| Task | Dice Score | Sensitivity | Specificity |
|------|-----------|-------------|-------------|
| Prostate Gland | 0.92 ± 0.03 | 94.5% | 98.2% |
| TZ/PZ Zones | 0.88 ± 0.05 | 91.3% | 96.8% |
| csPCa Detection | 0.76 ± 0.08 | 85.7% | 92.4% |

---

## 📁 Supported Formats

### Input Formats

- **DICOM** (`.dcm`, `.dicom`)
  - Single files or folder series
  - Automatic series grouping
  - Metadata preservation

- **NIfTI** (`.nii`, `.nii.gz`)
  - Compressed and uncompressed
  - Orientation handling (RAS/LPS)
  - Affine transformation support

- **MetaImage** (`.mha`, `.mhd`)
  - Header + raw data
  - Spacing and orientation metadata

- **NRRD** (`.nrrd`)
  - Medical research format
  - Full metadata support

### Output Formats

- **Segmentation Masks**: NIfTI, DICOM-SEG
- **Reports**: PDF, CSV, JSON
- **3D Models**: STL, OBJ (experimental)
- **Screenshots**: PNG, JPEG

---

## 📖 User Guide

### Patient Browser

**Purpose**: Load and manage medical images

**Features**:
- Multi-file selection
- Study/Series organization
- Metadata viewer
- Quick preview
- Recent files history

**Tips**:
- Use "Load DICOM Folder" for complete studies
- T2W sequences are automatically detected
- Cached images load faster on second access

### AI Analysis Panel

**Purpose**: Run automatic AI segmentation

**Workflow**:
1. Ensure AI models are loaded
2. Select analysis type
3. Choose T2W sequence from loaded cases
4. For csPCa: ADC and HBV are auto-detected
5. Click "Run AI Analysis"
6. Monitor progress bar
7. Review results with overlay

**Options**:
- Confidence threshold adjustment
- Batch processing (future)
- Custom model selection

### Manual Editing Panel

**Purpose**: Refine AI segmentations

**Tools**:
- **Brush**: Add/remove voxels
- **Eraser**: Quick removal
- **Fill**: Region filling
- **Interpolation**: Between slices

**Shortcuts**:
- `B`: Brush tool
- `E`: Eraser
- `Ctrl+Z`: Undo
- `Ctrl+Y`: Redo
- `+/-`: Adjust brush size

### Quantitative Analysis Panel

**Purpose**: Extract numerical measurements

**Metrics**:
- **Volume**: mm³, cm³, mL
- **Intensity**: Mean, median, std, min, max
- **Texture**: GLCM features, entropy
- **Shape**: Sphericity, compactness

**Export**:
- CSV format for Excel/Python
- Includes all ROI statistics
- Timestamp and patient metadata

### 3D Visualization

**Purpose**: Interactive 3D rendering

**Controls**:
- **Left Click + Drag**: Rotate
- **Right Click + Drag**: Pan
- **Scroll**: Zoom
- **R**: Reset view
- **W**: Wireframe mode
- **S**: Solid mode

---

## 🛠️ Development

### Project Structure

```
DeepProstate/
├── src/
│   ├── core/                    # Domain layer
│   │   ├── domain/
│   │   │   ├── entities/       # Medical entities
│   │   │   ├── repositories/   # Abstract repositories
│   │   │   ├── services/       # Domain services
│   │   │   └── value_objects/  # Immutable value objects
│   ├── use_cases/               # Application layer
│   │   └── application/
│   │       └── services/       # Use case implementations
│   ├── frameworks/              # Infrastructure layer
│   │   └── infrastructure/
│   │       ├── ui/             # PyQt6 widgets
│   │       ├── coordination/   # Orchestrators
│   │       ├── utils/          # Utilities
│   │       └── di/             # Dependency injection
│   └── adapters/                # External adapters
│       └── image_conversion/   # Format converters
├── resources/                   # UI resources
├── logs/                        # Application logs
├── medical_data/               # Patient data storage
├── config/                     # Configuration files
└── tests/                      # Unit tests
```

### Key Design Patterns

- **Dependency Injection**: Medical Service Container
- **Repository Pattern**: Data access abstraction
- **Service Layer**: Business logic encapsulation
- **Observer Pattern**: UI updates and events
- **Strategy Pattern**: Format conversion
- **Factory Pattern**: Widget creation
- **Singleton**: Global managers (cache, temp files)

### Coding Standards

```python
# Follow PEP 8
# Use type hints
def analyze_image(
    image: MedicalImage,
    analysis_type: AIAnalysisType
) -> SegmentationResult:
    """
    Analyze medical image using AI.

    Args:
        image: Input medical image
        analysis_type: Type of analysis to perform

    Returns:
        Segmentation result with masks and metadata

    Raises:
        ValueError: If image is invalid
        AIAnalysisError: If analysis fails
    """
    pass

# Use descriptive variable names
# Add docstrings to all public methods
# Log important operations
# Handle errors gracefully
```

### Running Tests

```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# Coverage report
python -m pytest --cov=src tests/
```

### Building from Source

```bash
# Create distribution
python setup.py sdist bdist_wheel

# Install locally
pip install -e .
```

## 🤝 Contributing

We welcome contributions from the medical imaging and AI community!

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Guidelines

- Follow existing code style and architecture
- Add tests for new features
- Update documentation
- Ensure all tests pass
- Add descriptive commit messages

### Areas for Contribution

- 🎯 Additional AI models (e.g., PI-RADS scoring)
- 📊 Advanced analytics and reporting
- 🌐 Multi-language support
- 🧪 Automated testing suite expansion
- 📚 Documentation improvements
- 🐛 Bug fixes and performance optimization

---

## 📄 License

This software is intended for **research and educational purposes** in medical imaging.

**Important**: This is **not FDA-approved** medical device software. Not intended for clinical diagnostic use without proper validation and regulatory clearance.

For commercial licensing inquiries, please contact: [your-email@domain.com]

---

## 📚 Citation

If you use DeepProstate in your research, please cite:

```bibtex
@software{deepprostate,
  title={DeepProstate: AI-Powered Prostate MRI Analysis Platform},
  author={Ronald Marca},
  year={2025},
  version={21.0},
  url={https://github.com/Marquita-oss/DeepProstate}
}
```

### Related Publications

- nnUNet: Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods (2021).

---

## 📞 Support

### Community

- **Issues**: [GitHub Issues](https://github.com/Marquita-oss/DeepProstate/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Marquita-oss/DeepProstate/discussions)
- **Email**: rnldmarca@gmail.org

### Reporting Bugs

Please include:
- OS and Python version
- Steps to reproduce
- Expected vs actual behavior
- Log files (from `logs/` directory)
- Screenshots if applicable

---

## 🙏 Acknowledgments

- **nnUNet Team**: For the excellent segmentation framework
- **PyQt6**: For the powerful UI framework
- **VTK Community**: For 3D visualization tools
- **pydicom**: For DICOM handling capabilities
- **Medical Imaging Community**: For valuable feedback

---

<p align="center">
  Made with ❤️ for the Medical Imaging Community
</p>

<p align="center">
  <strong>DeepProstate</strong> - Advancing Prostate Cancer Detection Through AI
</p>
