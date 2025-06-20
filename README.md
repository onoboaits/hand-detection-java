# Real-Time Hand Landmark Detection (Java + ONNX)

## Overview
This project is a real-time hand landmark detection system implemented in Java using an ONNX model. It captures video frames from a camera or video stream, processes the frames to detect hand landmarks, and displays the results in real time. This solution is ideal for applications such as gesture recognition, human-computer interaction, and augmented reality.

## Features
- Real-time hand landmark detection
- ONNX model inference using Java ONNX Runtime
- Supports webcam and video file input
- Visualizes detected hand landmarks on the video stream
- Modular and extensible design

## Getting Started

### Prerequisites
- Java 11 or later
- ONNX Runtime for Java
- OpenCV for Java

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/onoboaits/hand-detection-java.git
   cd hand-detection-java
   ```
2. Install the required dependencies (ONNX Runtime, OpenCV) via Maven or Gradle.

### Model
Download the ONNX hand landmark detection model and place it in the `models/` directory. Ensure that the model path is correctly configured in the `config.properties` file.

### Usage
Run the application:
```bash
mvn clean compile exec:java -Dexec.mainClass="com.yourcompany.HandLandmarkApp"
```

### Configuration
Edit the `config.properties` file to set:
- `video.source` (camera index or video file path)
- `model.path` (path to ONNX model)
- Other adjustable detection and rendering settings

## Demo
Insert a screenshot or gif here showing the hand landmark detection in action.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, enhancements, or new feature ideas.

## Contact
For questions or collaboration, contact: [onoboaits@gmail.com]


