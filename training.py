import pandas as pd
from pathlib import Path
from fastai import *
from fastai.vision import *

new_train_df = pd.read_csv("sample_train.csv")
image_train_path = Path("data/mono_images/train")
fnames = [image_train_path / path for path in list(new_train_df["fnames"])]
labels = new_train_df["target"]
path = Path("data")
bs=32
data = ImageDataBunch.from_lists(path, fnames, labels, ds_tfms=get_transforms(), bs=bs)
data.normalize(imagenet_stats)

# stage 1
learn = cnn_learner(data, models.resnet152, metrics=accuracy)
learn.fit_one_cycle(7, max_lr=1e-2)

# stage 2
data.batch_size = 16
learn.unfreeze()
lr = 1e-4
lrs = np.array([lr/9,lr/6,lr])
learn.fit_one_cycle(12, lrs)

# stage 3
learn.freeze()
lr = 1e-4
learn.fit_one_cycle(8, lr)