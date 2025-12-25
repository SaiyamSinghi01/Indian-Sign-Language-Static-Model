# Indian Sign Language – Static Gesture Recognition

This project implements a **Static Indian Sign Language (ISL) recognition system**
that **extracts hand landmarks from images using MediaPipe Hands** and trains
**deep learning models on the extracted landmark features** for accurate
real-time gesture classification.


## Processing Pipeline
1. Capture static hand images / webcam frames
2. Extract 21 hand landmarks using MediaPipe Hands
3. Normalize landmark coordinates
4. Train deep learning models on landmark vectors
5. Perform real-time gesture prediction


