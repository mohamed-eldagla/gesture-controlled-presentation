# Changelog

All notable changes to the PowerPoint Gesture Control Assistant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned Features
- macOS and Linux full support
- Multi-hand gesture support
- Google Slides integration
- Web-based interface
- Mobile companion app
- Cloud backup for user profiles
- Advanced analytics dashboard

## [1.0.0] - 2024-10-24

### Added
- Initial release of PowerPoint Gesture Control Assistant
- Hand gesture recognition using MediaPipe
- Facial recognition authentication using DeepFace
- Real-time emotion detection
- Custom gesture training system
- Camera calibration feature
- PowerPoint COM API integration (Windows)
- Keyboard simulation fallback
- Multi-tab user interface
- Voice feedback system
- User management system
- Settings configuration
- Comprehensive documentation
- Installation guide
- User guide
- Architecture documentation
- Contributing guidelines

### Built-in Gestures
- Open palm: Start presentation
- Index finger: Next slide
- Pinky finger: Previous slide
- Closed fist: End presentation

### Features
- Auto-pause on negative emotions
- Customizable gesture cooldown period
- Real-time status display
- Logging system
- Data persistence (users, gestures, calibration)

### Supported Platforms
- Windows 10/11 (full support with PowerPoint)
- macOS (keyboard simulation only)
- Linux (keyboard simulation only)

### Dependencies
- Python 3.8+
- OpenCV 4.8.0+
- MediaPipe 0.10.0+
- DeepFace 0.0.79+
- TensorFlow 2.15.0+
- PyAutoGUI 0.9.54+
- pyttsx3 2.90+
- comtypes 1.2.0+ (Windows only)

---

## Version History

### Version Numbering
- **Major version**: Breaking changes or significant new features
- **Minor version**: New features, backward compatible
- **Patch version**: Bug fixes and minor improvements

### Upgrade Guide
When upgrading between versions, please:
1. Backup your data files (users_db.pkl, custom_gestures.json)
2. Read the changelog for breaking changes
3. Update dependencies: `pip install -r requirements.txt --upgrade`
4. Test authentication and gestures after upgrade

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Suggesting features
- Contributing code
- Documentation improvements
