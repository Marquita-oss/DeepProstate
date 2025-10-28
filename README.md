# DeepProstate

<p align="center">
  <img src="deepprostate/resources/image/logo2.svg" alt="DeepProstate Logo" width="200"/>
</p>

<p align="center">
  <strong>AI-Powered Prostate MRI Analysis Platform</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/PyQt-6-green.svg" alt="PyQt6"/>
  <img src="https://img.shields.io/badge/AI-nnUNet-orange.svg" alt="nnUNet"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
</p>

---

## Overview

**DeepProstate** is a medical imaging application for prostate MRI analysis using AI-powered automatic segmentation with nnUNet v2. Built with Clean Architecture principles for reliability and maintainability.

### Key Features

- 🤖 **AI Segmentation**: Automatic prostate gland, zonal anatomy (TZ/PZ), and csPCa detection
- 🖼️ **Advanced Visualization**: Multi-planar views (Axial/Sagittal/Coronal) and 3D volume rendering
- ✏️ **Manual Editing**: Brush tools with undo/redo for segmentation refinement
- 📊 **Quantitative Analysis**: Volume calculations and radiomics metrics
- 🔄 **Format Support**: DICOM, NIfTI, MHA, NRRD
- 🛡️ **Medical Compliance**: HIPAA-compliant logging and audit trails

---

## Installation

### Requirements

- **Python**: 3.9+
- **RAM**: 8GB+ recommended
- **GPU**: NVIDIA GPU with CUDA (optional but highly recommended for AI inference)

### From PyPI (Recommended)

```bash
pip install deepprostate
```

### From Source

```bash
git clone https://github.com/Marquita-oss/DeepProstate.git
cd deepprostate
pip install -e .
```

### GPU Support (Recommended for AI Analysis)

For faster AI predictions, install PyTorch with CUDA support:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Note**: Without GPU, AI inference will be significantly slower (CPU-only mode).

### Verify Installation

```bash
deepprostate --version
```

---

## Quick Start

### Launch Application

```bash
deepprostate
```

### Basic Workflow

1. **Load AI Models**
   - Click "Load AI Models Path" in AI Analysis panel
   - Select folder containing nnUNet models

2. **Load Patient Data**
   - Use Patient Browser panel
   - Click "Load DICOM Folder" or "Load Single File"

3. **Run AI Analysis**
   - Select image in Patient Browser
   - Choose analysis type (Prostate/TZ-PZ/csPCa)
   - Click "Run AI Analysis"

4. **Review & Refine**
   - View results in 2D/3D viewers
   - Use Manual Editing tools to refine if needed
   - Export quantitative metrics

---

## AI Models

DeepProstate uses **nnUNet v2** for automatic segmentation:

| Model | Input | Output |
|-------|-------|--------|
| Prostate Gland | T2W | Complete prostate mask |
| Zonal Anatomy | T2W | TZ and PZ masks |
| csPCa Detection | T2W + ADC + HBV | Cancer lesion masks |

### Model Directory Structure

```
models/
├── Task500_ProstateGland/
│   └── nnUNetTrainer__nnUNetPlans__3d_fullres/
├── Task501_ProstateTZPZ/
│   └── nnUNetTrainer__nnUNetPlans__3d_fullres/
└── Task502_csPCa/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/
```

---

## Project Structure

```
deepprostate/
├── deepprostate/              # Main package
│   ├── core/                  # Domain layer
│   ├── use_cases/             # Application layer
│   ├── frameworks/            # Infrastructure layer
│   └── resources/             # UI resources
├── pyproject.toml             # Package configuration
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## License & Disclaimer

**MIT License** - See [LICENSE](LICENSE) file for details.

### Medical Disclaimer

⚠️ **IMPORTANT**: This software is intended for **research and educational purposes** only.

- **NOT** FDA-approved medical device software
- **NOT** intended for clinical diagnostic use
- **NOT** a substitute for professional medical judgment
- Users must obtain appropriate regulatory clearance for clinical use

---

## Citation

If you use DeepProstate in your research:

```bibtex
@software{deepprostate2025,
  title={DeepProstate: AI-Powered Prostate MRI Analysis Platform},
  author={Marca Ronald, Salas Rodrigo, Ponce Sebastian, Caprile Paola, Besa Cecilia},
  year={2025},
  version={1.4.0},
  url={https://github.com/Marquita-oss/DeepProstate}
}
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/Marquita-oss/DeepProstate/issues)
- **Email**: rnldmarca@gmail.com

---

## Acknowledgments

- **nnUNet Team**: Self-configuring segmentation framework
- **PyQt6**: UI framework
- **VTK**: 3D visualization
- **Medical Imaging Community**: Feedback and testing

---

<p align="center">
  Made with ❤️ for the Medical Imaging Community
</p>
