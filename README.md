# ADHD Audio Analysis System

A machine learning-based system for analyzing audio recordings to detect potential ADHD characteristics using advanced audio processing and machine learning techniques.

## Overview

This system uses audio processing and machine learning to analyze speech patterns and detect potential ADHD characteristics. It employs the eGeMAPs (extended Geneva Minimalistic Acoustic Parameter Set) feature set for audio analysis and uses Support Vector Machine (SVM) for classification.

## Features

- Audio file processing and segmentation
- eGeMAPs feature extraction
- Principal Component Analysis (PCA) for feature reduction
- SVM-based ADHD classification
- Real-time processing capabilities
- Web interface for easy interaction
- Support for MP3 and WAV audio formats

## Technical Stack

- Python 3.x
- Flask (Web Framework)
- Librosa (Audio Processing)
- OpenSMILE (eGeMAPs Feature Extraction)
- scikit-learn (Machine Learning)
- Pandas & NumPy (Data Processing)
- Matplotlib & Seaborn (Visualization)

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd ADHD
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the web server:

```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload an audio file (MP3 or WAV format) through the web interface

4. The system will process the audio and provide:
   - ADHD probability assessment
   - Classification result
   - Detailed analysis report

## Project Structure

```
ADHD/
├── app.py                 # Main Flask application
├── create_predict_data.py # Audio processing and feature extraction
├── predict.py            # ADHD prediction module
├── svm.py               # SVM model training and evaluation
├── pca.py              # PCA analysis and visualization
├── egemaps.py          # eGeMAPs feature extraction utilities
├── static/             # Static files (CSS, JS)
├── templates/          # HTML templates
├── uploads/            # Temporary storage for uploaded files
└── process/            # Processing directory for audio files
```

## Model Training

To train the model with your own dataset:

1. Place your audio files in the `dataset/train_16k` directory
2. Run the training script:

```bash
python create_train_test_data.py
```

## API Endpoints

- `GET /`: Main web interface
- `POST /upload_file`: Upload and process audio files
  - Accepts: MP3 or WAV files
  - Returns: JSON response with analysis results

## Performance

The system uses a combination of:

- eGeMAPs features for robust audio analysis
- PCA for feature reduction
- SVM with RBF kernel for classification
- Real-time processing capabilities

## License

This project is licensed under the terms of the included LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenSMILE for eGeMAPs feature extraction
- scikit-learn for machine learning capabilities
- Flask for web framework

#ADHD
#Pytorch
#Python
#eGeMAPs
