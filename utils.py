import numpy as np
from PIL import Image

import cv2
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    print(image_numpy.shape)
    cv2.imwrite(filename,image_numpy)
    # cv2.waitKey()
    # image_pil = Image.fromarray(image_numpy)
    # image_pil.save(filename)
    # print("Image saved as {}".format(filename))
