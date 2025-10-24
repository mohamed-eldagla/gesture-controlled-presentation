# Contributing to PowerPoint Gesture Control Assistant

First off, thank you for considering contributing to PowerPoint Gesture Control Assistant! It's people like you that make this project such a great tool.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and encourage diverse perspectives
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bug fix
4. Make your changes
5. Test your changes thoroughly
6. Push to your fork and submit a pull request

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples**
- **Describe the behavior you observed and what you expected**
- **Include screenshots or videos if possible**
- **Include your environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description of the proposed feature**
- **Explain why this enhancement would be useful**
- **Include mockups or examples if applicable**

### Your First Code Contribution

Unsure where to begin? You can start by looking through these issue labels:

- `good-first-issue` - Issues that should only require a few lines of code
- `help-wanted` - Issues that might be more involved
- `documentation` - Documentation improvements

### Pull Requests

- Fill in the required template
- Follow the style guidelines
- Include appropriate test cases
- Update documentation as needed
- End all files with a newline

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Webcam for testing
- Git

### Setup Steps

1. Clone your fork:
```bash
git clone https://github.com/your-username/powerpoint-gesture-control.git
cd powerpoint-gesture-control
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

### Running Tests

```bash
# Add test commands here when tests are implemented
python -m pytest tests/
```

### Running the Application

```bash
python main.py
```

## Pull Request Process

1. **Update Documentation**: Ensure any new features are documented in the README.md
2. **Update Requirements**: If you add dependencies, update requirements.txt
3. **Test Your Changes**: Make sure your changes work and don't break existing functionality
4. **Follow Coding Standards**: Ensure your code follows PEP 8 guidelines
5. **Write Clear Commit Messages**: Follow our commit message guidelines
6. **Create Pull Request**: Submit your PR with a clear title and description

### PR Review Process

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, a maintainer will merge your PR

## Style Guidelines

### Python Style Guide

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines:

- Use 4 spaces for indentation (not tabs)
- Line length should not exceed 100 characters
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Use type hints where appropriate

### Example:

```python
def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score between 0 and 1
    """
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)
```

### Documentation Style

- Use clear, concise language
- Include code examples where appropriate
- Keep documentation up to date with code changes
- Use proper markdown formatting

## Commit Message Guidelines

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, missing semi-colons, etc.)
- `refactor`: Code refactoring without changing functionality
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(gestures): add support for custom gesture combinations

Added ability to combine multiple finger positions for complex gestures.
This allows users to create more sophisticated control schemes.

Closes #123
```

```
fix(auth): resolve facial recognition timeout issue

Fixed issue where authentication would timeout on slower systems.
Increased timeout threshold and added better error handling.

Fixes #456
```

## Code Review Checklist

Before submitting your PR, ensure:

- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have docstrings
- [ ] No unnecessary comments or debug code
- [ ] All new features are documented
- [ ] Changes are tested on your local machine
- [ ] No merge conflicts with main branch
- [ ] Commit messages follow guidelines
- [ ] PR description clearly explains changes

## Areas for Contribution

We especially welcome contributions in these areas:

- **Cross-platform support**: macOS and Linux compatibility
- **Testing**: Unit tests and integration tests
- **Documentation**: Tutorials, examples, and guides
- **Accessibility**: Making the system more accessible
- **Performance**: Optimization and efficiency improvements
- **New Features**: Additional gesture recognition patterns
- **UI/UX**: Interface improvements
- **Internationalization**: Multi-language support

## Questions?

Don't hesitate to ask questions by opening an issue with the `question` label.

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes for their contributions
- Project documentation

Thank you for contributing to PowerPoint Gesture Control Assistant!
