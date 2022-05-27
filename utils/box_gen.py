import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import copy
from tqdm import tqdm
import os, glob

# ________ GENERATE BOX _________
from utils.preprocess import *
from utils.path_help import *

def gen_box(path_bg, path_fg):
    background = Image.open(path_bg)
    frontImage = Image.open(path_fg)

    frontImage = augment_image(frontImage)

    if background.width <= 640:
        frontImage = resize_pil(frontImage)

    width = np.random.randint(abs(background.width -  frontImage.width))
    height= (background.height - frontImage.height)//2

    img_merged = copy.deepcopy(background)
    img_merged.paste(frontImage, (width, height), frontImage)
    img_merged = np.array(img_merged)

    start_point = (width, height)
    end_point   = (width + frontImage.width, height + frontImage.height)

    # DEBUG
    # cv2.rectangle(img_merged, start_point, end_point, color, thickness)
    # imshow(img_merged, scale=0.8, verbose=True)
    return img_merged, [start_point, end_point, background.width, background.height]

def plot_yolo_box(img, label):
    Y, X = img.shape[0:2]
    img_class, x_c, y_c, w, h = label
    p1 = (int((x_c - w/2)*X), int((y_c - h/2)*Y))
    p2 = (int((x_c + w/2)*X), int((y_c + h/2)*Y))
    print(p1, p2)
    color = (255,0,0)
    thickness = 1
    return cv2.rectangle(img, p1, p2, color, thickness)

###______________GEN LABELS______________________

def yolo_format(class_index, point_1, point_2, width, height):
    # YOLO wants everything normalized
    # Order: class x_center y_center x_width y_height
    x_center = (point_2[0] + point_1[0]) / float(2*width)
    y_center = (point_2[1] + point_1[1]) / float(2*height)
    x_width = float(abs(point_2[0] - point_1[0])) / width
    y_height = float(abs(point_2[1] - point_1[1])) / height
    return str(class_index) + " " + str(x_center) \
       + " " + str(y_center) + " " + str(x_width) + " " + str(y_height)

def save_yolo_file(txt_path, line):
    with open(txt_path, 'w') as myfile:
        myfile.write(line + "\n") # append line

def dataframe_to_yolo_f(df, img_class=0, PATH='/content/dataset/labels'):
    n_files = len(glob.glob(PATH+'/*.txt'))
    for i in df.index:
        line = yolo_format(img_class, *df.iloc[i].values)
        save_yolo_file(f'{PATH}/img_{i+n_files}.txt', line)

def plt_imshow(img):
    plt.figure(dpi=100)
    plt.imshow(img)
    plt.grid(False)
    plt.axis(False)

def save_image(img_list, PATH = '/content/dataset/images'):
    n_files = len(glob.glob(PATH+'/*.jpg'))
    for i, img_ in enumerate(img_list):
        plt.imsave(f'{PATH}/img_{i+n_files}.jpg', img_)


# ______ FINAL FUNCTION _______
def generate_yolo_dataset(BACKGROUND_PATH, OBJECT_PATH, n_images=10, verbose=True):
    '''
    Create a full auto generation bounding box pipeline
    '''
    # Background
    all_background_dirs = get_dirs_sorted(BACKGROUND_PATH)
    background_dirs = all_background_dirs.sample(n_images)
    n_bg = len(all_background_dirs)

    # Object
    object_name  = OBJECT_PATH.split("/")[-2]
    all_object_dirs = get_dirs_sorted(OBJECT_PATH)

    # Verbose to Print
    if verbose:
        print(object_name)
        print("Number of all background images:", n_bg)

    # Create Output path
    create_path("runs")
    save_path = f"runs/{object_name}"
    create_path(save_path)
    create_path(f"{save_path}/images")
    create_path(f"{save_path}/labels")

    img_list, loc_list = [], []
    for idx, path_bg in enumerate(background_dirs):
        path_fg = all_object_dirs.sample(1).values[0]
        img_merged, loc = gen_box(path_bg, path_fg)

        # Save to list
        img_list.append(img_merged)
        loc_list.append(loc)

    # Save File
    save_image(img_list, PATH = save_path+"/images")
    df_loc = pd.DataFrame(loc_list, columns=['point_1', 'point_2', 'width', 'height'])

    dataframe_to_yolo_f(df_loc, img_class=object_name, PATH=save_path+"/labels")
    print(f"Finally Create {object_name}")