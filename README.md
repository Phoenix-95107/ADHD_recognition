# EMOVOCAL

<div align="center">
  <img src="static/img/logo.png" alt="EMOVOCAL Logo" width="200"/>
  <h3>Voice-Based ADHD Detection Using Machine Learning</h3>
  <p>An advanced web application leveraging voice analysis and machine learning for ADHD characteristic detection</p>
</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Deployment](#deployment)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

EMOVOCAL is a sophisticated Flask-based web application that performs real-time voice analysis for ADHD detection. The system processes audio recordings through state-of-the-art machine learning models trained on eGeMAPs (extended Geneva Minimalistic Acoustic Parameter Set) features to identify potential ADHD indicators in speech patterns.

## ✨ Features

- **Real-time Voice Analysis**

  - Instant processing and feedback
  - Support for MP3 and WAV audio formats
  - Dynamic progress tracking

- **Advanced Audio Processing**

  - Automated audio file segmentation
  - High-precision feature extraction using eGeMAPs
  - Multi-threaded processing for optimal performance

- **Machine Learning Integration**

  - State-of-the-art ADHD detection models
  - Probability-based classification
  - Continuous model improvement capabilities

- **User Interface**
  - Modern, responsive design
  - Intuitive user experience
  - Real-time progress visualization
  - Detailed results presentation

## 🛠 Technology Stack

- **Backend**

  - Python 3.13.2
  - Flask 3.1.0
  - OpenSMILE 2.5.1
  - scikit-learn 1.6.1

- **Audio Processing**

  - librosa 0.11.0
  - soundfile 0.13.1

- **Data Analysis**

  - pandas 2.2.3
  - numpy 2.1.3
  - matplotlib 3.10.1
  - seaborn 0.13.2

- **Development Tools**
  - joblib 1.3.2
  - tqdm 4.67.1

## 📁 Project Structure

```
.
├── app.py                 # Main Flask application
├── create_predict_data.py # Audio processing pipeline
├── predict.py            # ML model inference
├── static/              # Static assets
│   ├── css/            # Stylesheets
│   ├── js/             # JavaScript files
│   └── img/            # Images and icons
├── templates/          # HTML templates
│   ├── index.html     # Main interface
│   └── results.html   # Results display
├── uploads/           # Temporary file storage
└── requirements.txt   # Dependencies
```

## 📋 Prerequisites

- Python 3.13.2 or higher
- FFmpeg (for audio processing)
- gcc (for building dependencies)
- 4GB RAM minimum
- 2GB free disk space

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/Phoenix-ryan-1111/ADHD_recognition.git
cd EMOVOCAL
```

2. Create and activate virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix/MacOS
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Start the application:

```bash
python app.py
```

2. Access the web interface:

```
https://127.0.0.1:5000
```
Or

```
https://truthful-miracle-production.up.railway.app/
```

3. Upload an audio file (MP3/WAV)
4. Wait for analysis completion
5. Review detailed results

## 🔌 API Documentation

### Endpoints

- `GET /` - Main application interface
- `POST /upload_file` - Audio file upload endpoint
  - Accepts: multipart/form-data
  - Returns: Server-Sent Events (SSE) with analysis progress

### Response Format

```json
{
  "success": true,
  "prediction": "ADHD/Non-ADHD",
  "probability": "float (0-1)",
  "percentage": "float (0-100)"
}
```

## 🛠 Development

The application implements:

- Server-Sent Events for real-time updates
- Responsive design principles
- Modern CSS with flexbox/grid
- Event-driven JavaScript architecture

## 🚀 Deployment

Configured for Railway deployment with:

- Python 3.13.2 runtime
- Nixpacks builder
- Automatic environment configuration
- Production-grade server settings

## 🔒 Security Features

- Secure file upload handling
- File size restrictions (1000MB max)
- File type validation
- Secure filename processing
- Automated temporary file cleanup
- Production environment detection

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>Made with ❤️ by the EMOVOCAL Team</p>
</div>
