# MouthMap: Lip Reading with Deep Learning

<div align="center">
  <img src="Img-src\Lip Movement.gif" alt="MouthMap Logo">
</div>

## Exportred Model is Now Avaiable(Soon the HIGH Accuracy model will be Available : https://www.kaggle.com/models/santhankarnala/40th-epoch-model-checkpoint/Keras/default/1
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


## Using Exported Model
```py
import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List

def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :]) 
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    frames = tf.cast((frames - mean), tf.float32) / std 
    return frames

def load_alignments(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens.extend([' ', line[2]])
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str):
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    return frames, alignments

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz0123456789'?! "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def build_model():
    input_shape = (75, 46, 140, 1)
    inputs = tf.keras.layers.Input(shape=input_shape, name="input")
    
    x = tf.keras.layers.Conv3D(128, 3, activation=None, padding='same')(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(x)
    
    x = tf.keras.layers.Conv3D(256, 3, activation=None, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(x)
    
    x = tf.keras.layers.Conv3D(75, 3, activation=None, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(x)
    
    x = tf.keras.layers.Reshape((75, -1))(x)
    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(len(char_to_num.get_vocabulary()), activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

model = build_model()
model.load_weights('/kaggle/input/40th-epoch-model-checkpoint/keras/default/1/checkpoint.weights.h5')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=CTCLoss)

test_data = tf.data.Dataset.list_files(['data/s1/*.mpg']).map(lambda x: tf.py_function(load_data, [x], [tf.float32, tf.int64]))
test_data = test_data.padded_batch(2, padded_shapes=([75, None, None, 1], [None]))

videos = []
alignments = []
count = 0
for batch in test_data.take(3):
    batch_videos, batch_alignments = batch
    for i in range(len(batch_videos)):
        if count < 5: 
            videos.append(batch_videos[i:i+1]) 
            alignments.append(batch_alignments[i])
            count += 1
        else:
            break
    if count >= 5:
        break

for i in range(5):
    print(f"\nProcessing Video {i+1}:")
    video_input = videos[i]
    alignment = alignments[i]

    yhat = model.predict(video_input)


    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
    predicted_text = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode('utf-8')


    actual_text = tf.strings.reduce_join(num_to_char(alignment)).numpy().decode('utf-8')

    print("Actual:", actual_text)
    print("Predicted:", predicted_text)
    print("-" * 50)
```


## 🙏 Acknowledgments
- Built with TensorFlow
- Inspired by lip-reading research

---
