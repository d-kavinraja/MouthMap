import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from glob import glob
import cv2

class LipReadingDataGenerator(Sequence):
    def __init__(self, data_path, alignment_path, batch_size=32, frame_length=75,
                 image_height=46, image_width=140, **kwargs):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.alignment_path = alignment_path
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.image_height = image_height
        self.image_width = image_width

        self.video_paths = sorted(glob(os.path.join(data_path, '*.mpg')))
        self.alignment_paths = sorted(glob(os.path.join(alignment_path, '*.align')))

        print(f"Found {len(self.video_paths)} video files and {len(self.alignment_paths)} alignment files")
        self.vocabulary = self._create_word_vocabulary()

        self.char_to_num = tf.keras.layers.StringLookup(
            vocabulary=self.vocabulary, oov_token="")
        self.num_to_char = tf.keras.layers.StringLookup(
            vocabulary=self.vocabulary, oov_token="", invert=True)

    def _create_word_vocabulary(self):
        words = set()
        print(f"Processing alignment files from: {self.alignment_path}")

        for align_path in self.alignment_paths:
            try:
                with open(align_path, 'r') as f:
                    content = f.read().strip().split()
                    words.update([content[i] for i in range(2, len(content), 3)])
            except Exception as e:
                print(f"Error processing {align_path}: {str(e)}")

        words.discard('sil')
        vocabulary = sorted(list(words))

        if not vocabulary:
            print("No words found in alignment files. Using default vocabulary.")
            vocabulary = ['bin', 'blue', 'at', 'f', 'two', 'now']

        print(f"Vocabulary size: {len(vocabulary)}")
        return vocabulary

    def __len__(self):
        return max(1, len(self.video_paths) // self.batch_size)

    def _process_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mouth = gray[190:236, 80:220]
            mouth = cv2.resize(mouth, (self.image_width, self.image_height))
            frames.append(mouth)

        cap.release()

        frames = np.array(frames, dtype=np.float32)
        frames = (frames - frames.mean()) / (frames.std() + 1e-6)

        if len(frames) < self.frame_length:
            pad_length = self.frame_length - len(frames)
            frames = np.pad(frames, ((0, pad_length), (0, 0), (0, 0)), mode='constant')
        else:
            frames = frames[:self.frame_length]

        return frames

    def _process_alignment(self, alignment_path):
        with open(alignment_path, 'r') as f:
            content = f.read().strip().split()

        words = [content[i] for i in range(2, len(content), 3) if content[i] != 'sil']
        text = ' '.join(words)
        return self.char_to_num(tf.convert_to_tensor(text.split()))

    def __getitem__(self, idx):
        batch_videos = self.video_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_alignments = self.alignment_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = np.zeros((len(batch_videos), self.frame_length, self.image_height, self.image_width, 1))
        Y = np.zeros((len(batch_videos), len(self.vocabulary)))

        for i, (video_path, align_path) in enumerate(zip(batch_videos, batch_alignments)):
            frames = self._process_video(video_path)
            X[i] = frames.reshape(self.frame_length, self.image_height, self.image_width, 1)

            labels = self._process_alignment(align_path)
            Y[i] = tf.reduce_max(tf.one_hot(labels, len(self.vocabulary)), axis=0)

        return X, Y