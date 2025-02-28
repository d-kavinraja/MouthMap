import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from .data_generator import LipReadingDataGenerator  # Relative import

def predict_with_main_model(model_dir, video_path):
    vocab_path = os.path.join(model_dir, 'vocabulary_main.txt')
    with open(vocab_path, 'r') as f:
        vocabulary = f.read().splitlines()

    data_generator = LipReadingDataGenerator("", "") # Paths are not needed here for _process_video
    data_generator.vocabulary = vocabulary
    data_generator.char_to_num = tf.keras.layers.StringLookup(
        vocabulary=vocabulary, oov_token="")
    data_generator.num_to_char = tf.keras.layers.StringLookup(
        vocabulary=vocabulary, oov_token="", invert=True)

    model = load_model(os.path.join(model_dir, 'lip_reading_main_final.h5'))

    frames = data_generator._process_video(video_path)
    frames = frames.reshape(1, data_generator.frame_length,
                            data_generator.image_height,
                            data_generator.image_width, 1)

    prediction = model.predict(frames)
    predicted_indices = tf.argmax(prediction, axis=1).numpy()

    predicted_text = ' '.join([vocabulary[int(idx)] for idx in predicted_indices])

    return predicted_text

# Example usage
if __name__ == "__main__":
    model_directory = "models_main"
    test_video = r"data/s1/bbbf6n.mpg" 

    print("\nMaking predictions...")
    sentence_prediction = predict_with_main_model(model_directory, test_video)
    print(f"Predicted sentence: {sentence_prediction}")