# Mouth Map: Lip Reading to Sentence Conversion

## Overview
Mouth Map is a deep learning project aimed at converting lip movements into meaningful sentences. The project explores different approaches to training models for accurate lip reading and sentence generation. 
# DATASET : https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL

## Approaches
We have experimented with three different approaches to train the model:

### Approach 1: Baseline Model
- Utilized a CNN-LSTM hybrid model to extract lip movement features and predict words.
- Preprocessed data using OpenCV and Dlib.
- Achieved initial accuracy but required improvements in handling complex sentences.

### Approach 2: Transformer-Based Model
- Implemented a Vision Transformer (ViT) for feature extraction.
- Used an attention-based mechanism for improved sentence generation.
- Improved accuracy over Approach 1 but struggled with low-resolution inputs.

### Approach 3: 3D-CNN + BI-LSTM OR BI-GRU (Current Work)
- Currently exploring a model for better spatial feature extraction.
- Designed to enhance accuracy in real-time lip reading applications.
- Ongoing testing and optimization for robustness.

## Future Improvements
While working on the third approach, our teammates identified opportunities to refine all three approaches:
- **Approach 1:** Enhance feature extraction with additional layers or fine-tuning.
- **Approach 2:** Optimize transformer layers and experiment with different embedding techniques.
- **Approach 3:** Improve real-time performance and integrate with deployment frameworks.

## Contribution
We welcome contributions! If you have ideas for improving any of the approaches, feel free to raise an issue or submit a pull request.

## Contact
For any questions or discussions, please open an issue or reach out to us.

---

Stay tuned for updates as we refine the model and improve its accuracy!
