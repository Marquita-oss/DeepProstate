# Contributing to DeepProstate

Thank you for your interest in contributing to DeepProstate! This document provides guidelines for contributing to this medical imaging AI project.

## 🎯 Code of Conduct

### Our Standards

- **Patient Safety First**: All contributions must prioritize patient safety and data privacy
- **Professional Conduct**: Maintain respectful and constructive communication
- **Quality Over Speed**: Thorough testing is more important than quick delivery
- **Medical Ethics**: Follow medical software best practices and ethical guidelines

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/Marquita-oss/DeepProstate.git
cd deep-prostate
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv medical-env
source medical-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov pylint black mypy
```

### 3. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a bugfix branch
git checkout -b fix/bug-description
```

## 📝 Contribution Guidelines

### Code Style

#### Python Code Standards

```python
# Follow PEP 8
# Use type hints for all function signatures
def process_medical_image(
    image: MedicalImage,
    parameters: ProcessingParameters
) -> ProcessedResult:
    """
    Process medical image with given parameters.

    Args:
        image: Input medical image to process
        parameters: Processing configuration

    Returns:
        Processed result with metadata

    Raises:
        ValueError: If image is invalid
        ProcessingError: If processing fails
    """
    pass

# Use descriptive variable names
patient_age_years = 65  # Good
pa = 65  # Bad

# Add docstrings to all public methods
# Use type annotations
# Keep functions small and focused
```

#### Clean Architecture

Maintain the existing architecture layers:

```
core/domain/        # Pure business logic, no external dependencies
use_cases/          # Application-specific business rules
frameworks/         # UI, infrastructure, external libraries
adapters/           # Interface adapters and converters
```

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```bash
# Feature
feat(ai-analysis): add support for PI-RADS v2.1 scoring

# Bug fix
fix(dicom-loader): handle missing patient metadata gracefully

# Documentation
docs(readme): update installation instructions for Windows

# Refactoring
refactor(ui): extract common widget logic to base class

# Tests
test(segmentation): add unit tests for mask merging

# Performance
perf(rendering): optimize 3D volume rendering pipeline
```

### Testing Requirements

All contributions **must** include tests:

```python
# Unit tests for domain logic
def test_prostate_volume_calculation():
    """Test accurate volume calculation from segmentation mask."""
    mask = create_test_mask(shape=(100, 100, 20))
    spacing = ImageSpacing(x=0.5, y=0.5, z=3.0)

    volume = calculate_volume(mask, spacing)

    assert volume > 0
    assert volume < 1000  # Sanity check for prostate size
```

#### Test Coverage

- Aim for **80%+ code coverage**
- **100% coverage** for critical medical logic
- Include edge cases and error conditions

#### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_segmentation.py

# Run with verbose output
pytest -v
```

### Documentation

- Update **README.md** if adding user-facing features
- Add **docstrings** to all public classes and methods
- Update **API documentation** for interface changes
- Include **inline comments** for complex logic

## 🐛 Reporting Bugs

### Before Reporting

1. Check if the bug has already been reported
2. Verify it's reproducible in the latest version
3. Collect relevant information

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
1. Load DICOM file with...
2. Click on AI Analysis...
3. See error

**Expected Behavior**
What should have happened

**Screenshots**
If applicable

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python Version: [e.g., 3.10.5]
- DeepProstate Version: [e.g., v21.0]

**Logs**
Attach relevant log files from `logs/` directory

**Patient Data**
DO NOT include actual patient data.
Use anonymized test data only.
```

## ✨ Feature Requests

### Feature Request Template

```markdown
**Problem Statement**
What clinical/research problem does this solve?

**Proposed Solution**
Describe your proposed implementation

**Alternatives Considered**
Other approaches you've thought about

**Medical Use Case**
How will this benefit clinical workflow?

**Additional Context**
Any other relevant information
```

## 🔬 Medical Software Considerations

### HIPAA Compliance

- **Never log patient identifiable information**
- Use anonymized IDs in logs and error messages
- Encrypt sensitive data at rest
- Implement proper access controls

### Validation Requirements

For medical algorithms:
- Provide **validation methodology**
- Include **test datasets** (anonymized)
- Document **performance metrics** (sensitivity, specificity, etc.)
- Reference **published literature** if applicable

### Safety Checks

All medical analysis code must:
- **Validate inputs** thoroughly
- **Handle edge cases** gracefully
- **Provide confidence scores** when applicable
- **Log all operations** for audit trail

## 📋 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No patient data included
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Code reviewed locally

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change
- [ ] Documentation update

## Medical Impact
- [ ] No impact on clinical functionality
- [ ] Affects segmentation accuracy
- [ ] Changes visualization
- [ ] Modifies quantitative measurements

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style
- [ ] Self-review performed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests pass locally
```

### Review Process

1. **Automated Checks**: CI/CD runs tests and linting
2. **Code Review**: Maintainer reviews code quality
3. **Medical Review**: Clinical expert reviews medical accuracy (for algorithms)
4. **Approval**: 2+ approvals required for medical functionality

## 🏗️ Architecture Contributions

### Adding New Features

Follow Clean Architecture principles:

```python
# 1. Domain Entity (core/domain/entities/)
class NewMedicalEntity:
    """Pure business logic, no external dependencies."""
    pass

# 2. Repository Interface (core/domain/repositories/)
class NewEntityRepository(Protocol):
    """Abstract interface for data access."""
    def save(self, entity: NewMedicalEntity) -> None: ...

# 3. Use Case (use_cases/application/services/)
class NewFeatureService:
    """Application-specific business logic."""
    def __init__(self, repository: NewEntityRepository):
        self._repository = repository

# 4. Infrastructure (frameworks/infrastructure/)
class ConcreteRepository(NewEntityRepository):
    """Concrete implementation with external dependencies."""
    pass
```

### UI Contributions

For PyQt6 widgets:
- Follow existing widget patterns
- Separate UI logic from business logic
- Use signals/slots for communication
- Add keyboard shortcuts for common actions
- Ensure accessibility (screen readers, high contrast)

## 📞 Getting Help

- **GitHub Discussions**: Ask questions and discuss ideas
- **GitHub Issues**: Report bugs and request features
## 🎖️ Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Credited in academic publications (if significant contribution)

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

**Thank you for helping make DeepProstate better! 🙏**

Every contribution, no matter how small, helps improve prostate cancer detection and patient care.
