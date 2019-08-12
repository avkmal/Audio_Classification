import pandas as pd
from pathlib import Path
from fastai import *
from fastai.vision import *

test_df = pd.read_csv("test.csv")
new_train_df = pd.read_csv("sample_train.csv")
image_train_path = Path("data/mono_images/train")
fnames = [image_train_path / path for path in list(new_train_df["fnames"])]
labels = new_train_df["target"]
path = Path("data")
bs=32
data = ImageDataBunch.from_lists(path, fnames, labels, ds_tfms=get_transforms(), bs=bs)
data.normalize(imagenet_stats)

# add test data
path = Path("data/mono_images/test")
data.add_test(ImageList.from_df(test_df[["fnames", "target"]], path))


y_true = test_df["target"].map({'de': 0, "en": 1, "es": 2})
y_true = np.array(y_true)
y_true = torch.from_numpy(y_true)

learn = cnn_learner(data, models.resnet152, metrics=accuracy)
learn.load("final_model")
log_testpreds, _ = learn.get_preds(ds_type=DatasetType.Test)
print(f"accuracy: {accuracy(log_testpreds, y_true)}")
