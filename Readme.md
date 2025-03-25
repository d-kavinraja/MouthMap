# MouthMap: Lip Reading with Deep Learning

## Overview
MouthMap is a deep learning-based project designed to interpret lip movements from video data and generate corresponding text sentences. Leveraging convolutional neural networks (CNNs), bidirectional LSTMs, and Connectionist Temporal Classification (CTC) loss, this project processes video frames of lip movements to predict spoken phrases. It aims to assist in applications such as silent speech recognition, accessibility tools, and human-computer interaction.

## Features
- **Video Preprocessing**: Converts video frames to grayscale and normalizes them for model input.
- **Lip Reading Model**: Uses a 3D CNN and Bidirectional LSTM architecture to extract spatial-temporal features from lip movements.
- **CTC Loss**: Implements CTC loss for sequence-to-sequence prediction without explicit alignment.
- **Data Pipeline**: Efficiently handles video and alignment data using TensorFlow's tf.data API.
- **Real-Time Predictions**: Outputs predicted sentences from lip movement videos.

## Dataset
The project uses a dataset of lip movement videos (.mpg files) and corresponding text alignments (.align files). 

### Dataset Structure
- **Download Link**: [Google Drive URL](https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL)
- `data/s1/*.mpg`: Video files containing lip movements
- `data/alignments/s1/*.align`: Text alignments for the spoken phrases

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- gdown (for dataset downloading)

### Setup Steps

1. Clone the Repository:
```bash
git clone https://github.com/yourusername/MouthMap.git
cd MouthMap
```

2. Install Dependencies:
```bash
pip install -r requirements.txt
```

3. Download and Extract Dataset:
```python
import gdown
url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
output = 'data.zip'
gdown.download(url, output, quiet=False)
gdown.extractall('data.zip')
```

## Usage

### Training the Model
To train the MouthMap model:
```bash
python train.py
```
- Trains for 100 epochs
- Uses callbacks for checkpointing, learning rate scheduling, and example prediction
- Checkpoints saved in `models/checkpoint.weights.h5`

### Inference
To test the model on a single video:
```python
# Load trained model weights
model.load_weights('./models/checkpoint.weights.h5')

# Predict on a video
test_path = './data/s1/bbal6n.mpg'
frames, alignments = load_data(tf.convert_to_tensor(test_path))
yhat = model.predict(frames[tf.newaxis, ...])
decoded = tf.keras.backend.ctc_decode(yhat, [75], greedy=False)[0][0].numpy()
print(tf.strings.reduce_join([num_to_char(tf.argmax(x)) for x in yhat[0]]).numpy().decode('utf-8'))
```

### Example Output
- **Original**: "bin blue at l six now"
- **Prediction**: (varies based on training)

## Model Architecture

### Components
- **3D Convolutional Layers**:
  - Extracts spatial-temporal features
  - Input Shape: (75, 46, 140, 1)
  - 3 Conv3D layers with ReLU activation and MaxPooling3D

- **Bidirectional LSTMs**:
  - Two layers with 128 units each
  - 50% dropout
  - Captures temporal dependencies

- **Dense Layer**:
  - Outputs probabilities over vocabulary (41 characters including blank)

**Total Parameters**: ~8.47M

### Training Details
- **Optimizer**: Adam (learning rate = 0.0001)
- **Loss Function**: CTC Loss
- **Batch Size**: 2
- **Epochs**: 100

## Future Improvements
- Enhance model accuracy with larger datasets or data augmentation
- Add real-time video processing capabilities
- Optimize the model for deployment on edge devices
- Incorporate attention mechanisms for better sequence alignment

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License.

## Acknowledgments
- Dataset provided by [source if known]
- Built with TensorFlow
- Inspired by lip-reading research in deep learning

## Notes for Customization
- Replace `yourusername` in the clone URL with your actual GitHub username
- Add a `requirements.txt` file with exact package versions
- Consider adding a `train.py` script if not already present
