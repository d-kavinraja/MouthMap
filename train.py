import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from .data_generator import LipReadingDataGenerator  # Relative import
from .model import build_model  # Relative import

def train_and_save_main_model(data_dir, alignment_dir, batch_size=32):
    print("Starting training for sentence-level prediction...")

    model_dir = "models_main"
    os.makedirs(model_dir, exist_ok=True)

    data_generator = LipReadingDataGenerator(data_dir, alignment_dir, batch_size=batch_size)

    model = build_model(
        frame_length=75,
        image_height=46,
        image_width=140,
        vocabulary_size=len(data_generator.vocabulary)
    )

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(
            os.path.join(model_dir, 'lip_reading_main_best.keras'),
            save_best_only=True,
            monitor='accuracy'
        ),
        EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
    ]

    print("Training started...")
    model.fit(
        data_generator,
        epochs=10,
        callbacks=callbacks
    )

    final_model_path = os.path.join(model_dir, 'lip_reading_main_final.h5')
    model.save(final_model_path)
    print(f"Final model saved: {final_model_path}")

    vocab_path = os.path.join(model_dir, 'vocabulary_main.txt')
    with open(vocab_path, 'w') as f:
        f.write('\n'.join(data_generator.vocabulary))
    print(f"Vocabulary saved: {vocab_path}")

if __name__ == "__main__":
    data_dir = r"datas/s1"
    alignment_dir = r"datas/alignments/s1"

    print("Training main model...")
    train_and_save_main_model(data_dir, alignment_dir)