import numpy as np
np.random.seed(1001)

import os
import shutil
import soundfile as sf
import IPython
import matplotlib
import pandas as pd
import seaborn as sns
import librosa
import librosa.display
import re
import time
from pathlib import Path
from itertools import islice
from IPython.display import Audio
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import get_window
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm_notebook as tqdm

## Convert to mono

DATA = Path('data')

TRAIN = DATA/'train'
TEST = DATA/'test'

MONO_TRAIN = DATA/'mono_train'
MONO_TEST = DATA/'mono_test'

def read_file(filename, path='', sample_rate=None, trim=False):
    ''' Reads in a flac file and returns it as an np.float32 array in the range [-1,1] '''
    filename = Path(path) / filename
    data, file_sr = librosa.load(filename)
    if data.dtype == np.int16:
        data = np.float32(data) / np.iinfo(np.int16).max
    elif data.dtype != np.float32:
        raise OSError('Encounted unexpected dtype: {}'.format(data.dtype))
    if sample_rate is not None and sample_rate != file_sr:
        if len(data) > 0:
            data = librosa.core.resample(data, file_sr, sample_rate, res_type='kaiser_fast')
        file_sr = sample_rate
    if trim and len(data) > 1:
        data = librosa.effects.trim(data, top_db=40)[0]
    return data, file_sr

def write_file(data, filename, path='', sample_rate=44100):
    ''' Writes a flac file to disk stored as int16 '''
    filename = Path(path) / filename
    if data.dtype == np.int16:
        int_data = data
    elif data.dtype == np.float32:
        int_data = np.int16(data * np.iinfo(np.int16).max)
    else:
        raise OSError('Input datatype {} not supported, use np.float32'.format(data.dtype))
    sf.write(filename, int_data, sample_rate)
    
def _to_mono(filename, dest_path):
    data, sr = read_file(filename)
    if len(data.shape) > 1:
        data = librosa.core.to_mono(data.T) # expects 2,n.. read_file returns n,2
    write_file(data, dest_path/filename.name, sample_rate=sr)


def convert_to_mono(src_path, dest_path, processes=None):
    src_path, dest_path = Path(src_path), Path(dest_path)
    os.makedirs(dest_path, exist_ok=True)
    filenames = list(src_path.iterdir())
    convert_fn = partial(_to_mono, dest_path=dest_path)
    with Pool(processes=processes) as pool:
        with tqdm(total=len(filenames), unit='files') as pbar:
            for _ in pool.imap_unordered(convert_fn, filenames):
                pbar.update()
                
convert_to_mono(TRAIN, MONO_TRAIN)

convert_to_mono(TEST, MONO_TEST)




## Generate images

DATA = Path('data')

# these folders must be in place
TRAIN_PATH = DATA/'mono_train'
TEST_PATH = DATA/'mono_test'

# these folders will be created
IMAGES = DATA/'mono_images'
TRAIN_IMAGE_PATH = IMAGES/'train'
TEST_IMAGE_PATH = IMAGES/'test'

train_fnames = [f.name for f in TRAIN_PATH.iterdir()]
test_fnames = [f.name for f in TEST_PATH.iterdir()]

def log_mel_spec_tfm(fname, src_path, dst_path):
    x, sample_rate = read_file(fname, src_path)
    
    n_fft = 1024
    hop_length = 256
    n_mels = 40
    fmin = 20
    fmax = sample_rate / 2 
    
    mel_spec_power = librosa.feature.melspectrogram(x, sr=sample_rate, n_fft=n_fft, 
                                                    hop_length=hop_length, 
                                                    n_mels=n_mels, power=2.0, 
                                                    fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)
    dst_fname = dst_path / (".".join(fname.split('.')[:-1]) + '.png')
    plt.imsave(dst_fname, mel_spec_db)
    
def transform_path(src_path, dst_path, transform_fn, fnames=None, processes=None, delete=False, **kwargs):
    src_path, dst_path = Path(src_path), Path(dst_path)
    if dst_path.exists() and delete:
        shutil.rmtree(dst_path)
    os.makedirs(dst_path, exist_ok=True)
    
    _transformer = partial(transform_fn, src_path=src_path, dst_path=dst_path, **kwargs)
    if fnames is None:
        fnames = [f.name for f in src_path.iterdir()]
    with Pool(processes=processes) as pool:
        with tqdm(total=len(fnames), unit='files') as pbar:
            for _ in pool.imap_unordered(_transformer, fnames):
                pbar.update()
                
transform_path(TRAIN_PATH, TRAIN_IMAGE_PATH, log_mel_spec_tfm, 
               fnames=train_fnames, delete=True)

transform_path(TEST_PATH, TEST_IMAGE_PATH, log_mel_spec_tfm, 
               fnames=test_fnames, delete=True)


## generate df for train and test

def get_filename(path):
    absolute_fname = path.as_posix()
    absolute_fname_parts = absolute_fname.split('/')
    fname = absolute_fname_parts[len(absolute_fname_parts) - 1]
    return fname

fnames = [get_filename(path) for path in fnames]

audio_train_path = Path("data/mono_images/train")
audio_train_paths = audio_train_path.ls()

# extract data from filenames

fnames = [get_filename(path) for path in audio_train_paths]
target = [f[:2] for f in fnames]
gender = [f[3] for f in fnames]
pattern = re.compile(r"speed[0-9]*|noise[0-9]*|pitch[0-9]*")
transformation = [pattern.findall(f) for f in fnames]
transformation = [f[0] if f else None for f in transformation]
pattern = re.compile(r"fragment[0-9]*")
fragment = [pattern.findall(f) for f in fnames]
fragment = [f[0] if f else None for f in fragment]

train_df = pd.DataFrame()
train_df["fnames"] = fnames
train_df["target"] = target
train_df["gender"] = gender
train_df["transformation"] = transformation
train_df["fragment"] = fragment

train_df.to_csv("train.csv")

audio_test_path = Path("data/mono_images/test")
audio_test_paths = audio_test_path.ls()

# extract data from filenames
fnames = [get_filename(path) for path in audio_test_paths]
target = [f[:2] for f in fnames]
gender = [f[3] for f in fnames]
pattern = re.compile(r"speed[0-9]*|noise[0-9]*|pitch[0-9]*")
transformation = [pattern.findall(f) for f in fnames]
transformation = [f[0] if f else None for f in transformation]
pattern = re.compile(r"fragment[0-9]*")
fragment = [pattern.findall(f) for f in fnames]
fragment = [f[0] if f else None for f in fragment]

test_df = pd.DataFrame()
test_df["fnames"] = fnames
test_df["target"] = target
test_df["gender"] = gender
test_df["transformation"] = transformation
test_df["fragment"] = fragment

audio_test_path = Path('data/test')
image_test_path = Path('data/test_spectrogram')

test_df.to_csv("test.csv")