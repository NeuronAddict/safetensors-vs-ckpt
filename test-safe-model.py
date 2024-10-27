from pathlib import Path

from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.external import untar_data, URLs
from fastai.data.transforms import get_image_files, parent_label, GrandparentSplitter
from fastai.metrics import error_rate
from fastai.vision.data import ImageBlock
from fastai.vision.learner import vision_learner
from matplotlib import image as mpimg, pyplot as plt
from safetensors.torch import load_model
from torchvision.models import resnet34

path = untar_data(URLs.MNIST)
datablock: DataBlock = DataBlock(
    blocks=[ImageBlock, CategoryBlock],
    get_items=get_image_files,
    get_y=parent_label,
    splitter=GrandparentSplitter(train_name='training', valid_name='testing')
)
dls = datablock.dataloaders(path, bs=256)
learn = vision_learner(dls, resnet34, metrics=error_rate)


load_model(learn, 'digits.safetensors')


digit_path = Path('tests')

images = get_image_files(digit_path, False)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, image in enumerate(images):
    row = i // 3
    col = i % 3

    pred = learn.predict(image)

    img = mpimg.imread(image)

    axes[row, col].imshow(img)
    axes[row, col].axis('off')
    axes[row, col].set_title(str(pred[0]))
plt.show()