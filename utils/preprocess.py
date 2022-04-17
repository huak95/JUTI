import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import ImageOps

def imshow(img_path, scale=0.4, verbose = False):
    if type(img_path) == str:
        img = cv2.imread(img_path)
    else:
        img = img_path
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = resize_cv2(img, scale)
    # img = cv2.resize(img)
    if verbose == True:
        # print(img_path)
        print('img_origin: ', img.shape)
        print('img_resize: ', img_resize.shape)
    plt.imshow(img_resize)
    plt.axis(False)
    plt.show()

def resize_cv2(img, scale=0.4):
    width = int(img.shape[1]*scale)
    height= int(img.shape[0]*scale)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def resize_pil(img, scale=0.4):
    width = int(img.width*scale)
    height= int(img.height*scale)
    dim = (width, height)
    return img.resize(dim)

def imsize(imgpath):
    img = cv2.imread('/content/extra/'+ imgpath, 0)
    return img.shape

def augment_image(img):
    if np.random.randint(2):
        img = ImageOps.mirror(img)
    img = img.rotate(np.random.randint(-5,5))
    img_scale = np.random.randint(80,120)/100
    img = resize_pil(img, scale=img_scale)
    return img

# def equalize(im):
#     h = im.convert("L").histogram()
#     lut = []
#     for b in range(0, len(h), 256):
#         # step size
#         step = reduce(operator.add, h[b:b+256]) / 255
#         # create equalization lookup table
#         n = 0
#         for i in range(256):
#             lut.append(n / step)
#             n = n + h[i+b]
#     # map image through lookup table
#     return im.point(lut*im.layers)