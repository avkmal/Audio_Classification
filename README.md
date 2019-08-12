# Spoken Language Identification

The task is to classify language spoken from audio files. The dataset is from https://www.kaggle.com/toponowicz/spoken-language-identification

For this problem I decides to take the approach of generating images of spectogram for each audio file then feed the images as input into convolutional neural network to classify the audio. Another approach is to just used the input frequency of the data as input for simple neural network. The other approach is to use neural network to convert audio to text then classify the language based on the text. I've decided to use first approach as it can give quite good accuracy.

I'll be using fastai brilliant library for modelling and librosa to preprocess the audio.

## Preprocessing Steps

- convert to mono to compress the data by only using single channel instead of stereo.
- Using short-time Fourier Transform(STFT) to plot audio spectrograms. This allows us to see how different frequencies change over time.
- convert to mel scale to convert regular spectrogram into melspectrogram.
- generate images of melspectrograms to feed into CNN.

## Modelling

- Using ResNet152 architecture.
- Training in 3 stage:
    - Stage 1:
        - initialize the model(resnet152 with pretrained weights on ImageNet).
        - train final layer
        - best learning rate: 0.01
        - number of epochs: 7
        - validation accuracy: 0.945
    - Stage 2:
        - unfreeze all layers and train.
        - best learning rate: 0.0001
        - number of epochs: 12
        - using differential learning rate: [lr/9, lr/6, lr]. The earlier layers was trained on very small learning rate, 0.001/9, the middle layers trained on a bit higher learning rate, 0.001/6, and final layers trained using learning rate, 0.001
        - validation accuracy: 0.0.976
    - Stage 3:
        - freeze the layers and train the final layer.
        - best learning rate: 0.0001
        - number of epochs: 8
        - validation accuracy: 0.0.986

- Accuracy on test set: **0.9463**

## Dependencies

- First download the data from https://www.kaggle.com/toponowicz/spoken-language-identification
- You need `fastai`, `librosa`, and `soundfile` library installed.
- run `preprocessing.py`, then `training.py`, and finally `testing.py`


## TO DO
- use virtual environment.
- generate requirements.txt



