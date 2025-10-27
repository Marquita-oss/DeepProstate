# Changelog

All notable changes to DeepProstate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2024-10-16

### Added
- Complete Clean Architecture
- AI-powered prostate segmentation using nnU-Net v2
- Multi-sequence MRI support (T2W, ADC, DWI, High B-Value)
- Quad-view medical image viewer with synchronized cross-hairs
- 3D volume rendering using VTK
- Patient browser with DICOM series organization
- AI analysis panel with multiple analysis types:
  - Prostate gland segmentation
  - Transition Zone (TZ) and Peripheral Zone (PZ) segmentation
  - Clinically Significant Prostate Cancer (csPCa) detection
- Manual editing tools for segmentation refinement
- Quantitative analysis with volume and intensity measurements
- Multi-format support: DICOM, NIfTI, MHA, NRRD
- HIPAA-compliant logging and audit trail
- Clean Architecture implementation
- PyQt6-based modern UI with dark theme support
- Comprehensive error handling and validation
- Resource management for images and icons
- Configuration system for AI models and application settings

### Changed
- Migrated from version-specific naming to clean branding
- Unified logging system across all modules
- Improved performance for image loading and caching

### Fixed
- Performance optimizations for quad view rendering
- Slice interpolation caching issues
- Cross-platform path handling for resources

## [Unreleased]

### Planned
- Batch processing for multiple studies
- Advanced radiomics feature extraction
- Export to DICOM-SEG format
- Integration with PACS systems
- Multi-language support
- Enhanced 3D visualization controls
- PI-RADS scoring assistant
- Report generation in PDF format

---

[1.4.0]: https://github.com/ronaldmarca/deepprostate/releases/tag/v1.4.0
