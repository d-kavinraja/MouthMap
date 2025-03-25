# MouthMap: Lip Reading with Deep Learning

<div align="center">
  <img src="Img-src\Lip Movement.gif" alt="MouthMap Logo">
</div>

## Overview

MouthMap is a deep learning-based project designed to interpret lip movements from video data and generate corresponding text sentences. Leveraging convolutional neural networks (CNNs), bidirectional LSTMs, and Connectionist Temporal Classification (CTC) loss, this project processes video frames of lip movements to predict spoken phrases.

### 🎯 Key Applications
- Silent speech recognition
- Accessibility tools
- Human-computer interaction

## 🌟 Features

- **Video Preprocessing**: Converts video frames to grayscale and normalizes them for model input
- **Lip Reading Model**: 
  - Uses 3D CNN and Bidirectional LSTM architecture
  - Extracts spatial-temporal features from lip movements
- **CTC Loss Implementation**: Sequence-to-sequence prediction without explicit alignment
- **Efficient Data Pipeline**: Handles video and alignment data using TensorFlow's tf.data API
- **Real-Time Predictions**: Outputs predicted sentences from lip movement videos

## 📦 Dataset

### Download Link
[Google Drive Dataset](https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL)

### Structure
- `data/s1/*.mpg`: Video files containing lip movements
- `data/alignments/s1/*.align`: Text alignments for spoken phrases

## 🚀 Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- gdown (for dataset downloading)

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/MouthMap.git
   cd MouthMap
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**
   ```python
   import gdown
   url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
   output = 'data.zip'
   gdown.download(url, output, quiet=False)
   gdown.extractall('data.zip')
   ```

## 🏋️ Model Architecture

### Components
- **3D Convolutional Layers**
  - Input Shape: (75, 46, 140, 1)
  - 3 Conv3D layers with ReLU activation and MaxPooling3D
- **Bidirectional LSTMs**
  - Two layers with 128 units
  - 50% dropout
- **Dense Layer**
  - Outputs probabilities over 41-character vocabulary

![Model Architecture](./Img-src/Model%20Architecture.png)

## 🔬 Training Details

- **Optimizer**: Adam (learning rate = 0.0001)
- **Loss Function**: CTC Loss
- **Batch Size**: 2
- **Epochs**: 100

### Callbacks
- ModelCheckpoint
- LearningRateScheduler
- ProduceExample

## 🧪 Inference Example

```python
# Load trained model weights
model.load_weights('./models/checkpoint.weights.h5')

# Predict lip movements
test_path = './data/s1/bbal6n.mpg'
frames, alignments = load_data(tf.convert_to_tensor(test_path))
yhat = model.predict(frames[tf.newaxis, ...])
decoded = tf.keras.backend.ctc_decode(yhat, [75], greedy=False)[0][0].numpy()
```

### Example Output
- **Original**: "bin blue at l six now"
- **Prediction**: *(varies based on training)*

## 🚧 Future Improvements
- Enhance model accuracy with larger datasets
- Add real-time video processing
- Optimize for edge device deployment
- Incorporate attention mechanisms

## 🤝 Contributing

Contributions are welcome! 
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open a Pull Request


## 🙏 Acknowledgments
- Built with TensorFlow
- Inspired by lip-reading research

---

