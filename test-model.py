from fastai.data.transforms import get_image_files
from fastai.learner import load_learner
from matplotlib import image as mpimg, pyplot as plt

from pathlib import Path



digit_path = Path('tests')
learn = load_learner('export.pkl')

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