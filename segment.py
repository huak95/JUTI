from ctypes import resize
from turtle import back, width
from transformers import pipeline
import torch
from utils import path_help
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.box_gen import *
from utils.path_help import *

from collections import Counter
from PIL import Image, ImageOps


def imshow_plt(pil_image):
    plt.imshow(resize_cv2(np.array(pil_image)), cmap='gray')

def get_object_ratio(masked_images):
    d = np.array(masked_images.getdata())
    d_count = Counter((d))
    object_ratio = d_count[255]/float(len(np.array(d)))
    return object_ratio

def treshold_segment(seg_image_in):
    object_ratio_array = []
    seg_image = []
    
    for si in seg_image_in:
        object_ratio = get_object_ratio(si['mask'])
        if object_ratio > 0.11:
            object_ratio_array.append(object_ratio)
            seg_image.append(si)

    return seg_image, object_ratio_array

def visualize_segment(image, seg_image_in):

    seg_image, object_ratio_array = treshold_segment(seg_image_in)

    with plt.style.context('classic'):

        plt.figure(figsize=(10, 15), dpi=100)
        n_images = len(seg_image) + 1

        plt.subplot(1, n_images, 1)
        title = "Original"
        imshow_plt(image)
        plt.title(title)
        plt.axis('off')

        for idx in range(1, n_images):
            plt.subplot(1, n_images, idx+1)
            img = seg_image[idx-1]["mask"]
            title = seg_image[idx-1]["label"]
            imshow_plt(img)
            plt.title(f"{title}-{object_ratio_array[idx-1]:.2f}")
            plt.axis('off')

        plt.show()
    return seg_image


def crop_resize(im, desired_size):
    width, height = im.size
    left = (width-height)//2
    right= left+height
    top = 0
    bottom = height

    im = im.crop((left,top,right,bottom))
    im = im.resize((desired_size, desired_size))
    return im
    
def resize_background(dirs_list: list, desired_size=500):
    im_list = []
    for dir in (dirs_list):
        im = Image.open(dir)
        im_crop = crop_resize(im, desired_size)
        im_list.append(im_crop)
    return im_list

def resize_object(im, desired_size=150):
    width, height = im.size
    if width > height:
        ratio = desired_size/width
    else:
        ratio = desired_size/height
    ratio = round(ratio, 2)
    im_resize = resize_pil(im, ratio)
    return im_resize
    
class FindLocation:
    def __init__(self):
        # model = "nvidia/segformer-b0-finetuned-ade-512-512"
        model = "segformer"
        # model = "facebook/detr-resnet-50-panoptic"
        self.model = pipeline("image-segmentation", model=model, device=0)
        pass
    
    def segment_from_file(self, file):
        pass
    
    def get_free_mem(self):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = t-r  # free inside reserved
        return f

    def get_batch_size(self):
        torch.cuda.empty_cache()
        a = self.get_free_mem()
        # a = torch.cuda.memory_reserved(0)
        batch_size = round((a * 6.34e-09 + -0.364)/2) 
        print("free mem:", a)
        print("batch size:", (batch_size))
        return batch_size
    
    
    def segment_from_list(self, dirs_list: list, verbose=True):
        all_labels = set()
        
        # path_help.create_path("data_resized")
        pil_list = resize_background(dirs_list, 500)

        seg_all_images = self.model(pil_list, batch_size=self.get_batch_size(), truncation="only_first")

        for i in range(len(seg_all_images)):
            img_path = dirs_list[i]
            seg_image = seg_all_images[i]

            # seg_image = model(img_path)
            image = Image.open(img_path)
            # image
            
            if verbose:
                print(img_path)
                seg_image_reduce = visualize_segment(image, seg_image)
            else:
                seg_image_reduce, object_ratio_array  = treshold_segment(seg_image)
                
            new_labels = set([si["label"] for si in seg_image_reduce])
            all_labels = all_labels.union(new_labels)
            
        return seg_all_images, list(all_labels)

    def pad_and_erode(self, mask_img):
        '''
        to padding the images
        '''
        kernel = np.ones((20,20), np.uint8)
        
        mask_flat = mask_img.ravel()
        
        if len(mask_flat[mask_flat == 0]) > len(mask_flat)*0.2:
            mask_img = np.pad(mask_img, 10) 
            mask_img = cv2.erode(mask_img, kernel, iterations=5)
            mask_img = mask_img[10:-10, 10:-10]
        return mask_img

    # Random point
    def random_point(self, mask_img, point=100, shrink=True):
        '''
        to get random coordinate from mask
        '''
        img_dilate = mask_img.copy()
        img_dilate = np.array(img_dilate, dtype=np.uint8)
        
        if shrink:
            img_dilate = self.pad_and_erode(img_dilate)
            
        img_bool = np.array(img_dilate, dtype=bool)
        x_ = np.argwhere(img_bool.ravel()).ravel()
        # print(x_)
        # print(x_.shape)
        sel_index = np.random.choice(x_, size=point)
        random_x, random_y = np.unravel_index(sel_index, img_bool.shape)
        return random_x, random_y
    
    def get_sample_labels(self, n_sample_percent=0.1, background_dirs="data/background/"):
        all_background_path = path_help.get_dirs_sorted(background_dirs)
        sample_size = int(len(all_background_path)*n_sample_percent)
        print("sample_size:", sample_size)
        all_labels = set()
        
        sample_background_path = all_background_path.sample(sample_size).to_list()
        seg_all, _ = self.segment_from_list(sample_background_path, False)
        
        print("Finished Pre-Segmentation")
        for i in range(len(seg_all)):
            background_path = sample_background_path[i]
            seg_image = seg_all[i]
            seg_image_reduce, _ = treshold_segment(seg_image)
            
            new_labels = set([si["label"] for si in seg_image_reduce])
            all_labels = all_labels.union(new_labels)
        
        self.all_labels = all_labels
        return all_labels
            

class generate_data():

    def __init__(self, all_labels, test=False):
        fg_labels = sorted(os.listdir("data/object"))
        bg_labels = all_labels.copy()
        select_bg_labels = []

        if test:
            select_bg_labels = [['road', 'floor', 'earth'], ['wall', 'building', 'house'],['road', 'floor', 'earth'],['road', 'floor', 'earth'],['road', 'floor', 'earth']]
            
        else:
            for fg_lab in fg_labels:
                # print(fg_lab)
                out = input(f"Selected bg label that '{fg_lab}' will fitted in [{bg_labels}]")
                select_bg_labels.append(out.split(','))
            
        self.select_bg_labels = select_bg_labels
        self.fg_label = fg_labels
        self.bg_label = bg_labels
        self.selected_df = pd.DataFrame(zip(fg_labels, select_bg_labels), columns=['fg','bg'])
        self.fl = FindLocation()
        self.bg_size = 500
        self.obj_size = 150
    
    def get_selected_df(self):
        return self.selected_df
    
    def get_fg_label(self, path_fg):
        object_name  = path_fg.split("/")[-2]
        return object_name
    
    def merge_mask_pil(self, seg_images, intersec_class_idx):
        pre_mask = seg_images[intersec_class_idx[0]]['mask']

        for idx in intersec_class_idx:
            mask = seg_images[idx]['mask']
            pre_mask = Image.blend(pre_mask, mask, .5)
        
        pre_mask = np.array(pre_mask, dtype='uint8')
        pre_mask[pre_mask > 0] = 255 # Set all values to 255
        return pre_mask
    
    def cal_mask_percent(self, mask):
        mask_flat = mask.ravel()
        mask_area = mask_flat[mask_flat > 0] 
        mask_percent = len(mask_area) / len(mask_flat)
        return mask_percent
    
    def set_background_size(self, im_size: int):
        self.bg_size = im_size
        print(f"Set Background Size to {im_size} px")

    def set_object_size(self, im_size: int):
        self.obj_size = im_size
        print(f"Set Object Size to {im_size} px")
    
    def gen_box_beta(self, seg_images, path_bg, path_fg, verbose=False):

        # Get variable from self
        object_name = self.get_fg_label(path_fg)
        sel_df = self.selected_df
        bg_class = sel_df[sel_df.fg == object_name].bg.values[0]
        
        if verbose:
            print("object_name:", object_name)
            print("bg_class:", bg_class)
        
        label_images = [si['label'] for si in seg_images]
        label_images = pd.DataFrame({"label_images": label_images})
        intersec_bg_class = label_images[label_images["label_images"].isin(bg_class)]
        # print(intersec_bg_class)
        intersec_class_idx = np.array(intersec_bg_class.index, dtype=int)
        
        if verbose:
            print("label_images", ",".join(label_images.iloc[:,0].to_list()))
            print("intersec_class_idx", intersec_class_idx)
            # print("label_images", label_images)
            
        # Skip when no label
        if len(intersec_class_idx) == 0:
            print("No Select Label -> Skip")
            return None, None
        
        # Load Images from path
        # background = Image.open(path_bg)
        # Resize and crop all images
        background = resize_background([path_bg], desired_size=self.bg_size)[-1]
        frontImage = Image.open(path_fg)
        frontImage = resize_object(frontImage, desired_size=self.obj_size)
        frontImage = augment_image(frontImage)
        
        if len(intersec_class_idx) == 1:
            mask_array = seg_images[intersec_class_idx[0]]['mask']
            mask_array = np.array(mask_array, dtype='uint8')
        else:
            mask_array = self.merge_mask_pil(seg_images, intersec_class_idx)
        
        # Calculate Percentage
        mask_unique = np.unique(mask_array.reshape(-1))
        mask_area_percentage = self.cal_mask_percent(mask_array)
        
        if len(mask_unique) != 1 :
            if mask_area_percentage > 0.4:
                shrink = True
            else:
                shrink = False    
            pil_mask = Image.fromarray(mask_array)

            # Randomize
            y, x = self.fl.random_point(mask_array, 100, shrink=shrink)
        else:
            print("No Select Label -> Skip")
            return None, None

        # if background.width <= 640:
            # frontImage = resize_pil(frontImage)

        # Get Centroid
        x_c = frontImage.width//2
        y_c = frontImage.height//2

        # Get width and Height of object images
        width = x[0] - x_c
        height = y[0]- y_c*2
        

        # Paste image                    
        img_merged = copy.deepcopy(background)
        img_merged.paste(frontImage, (width, height), frontImage)
        img_merged = np.array(img_merged)
        
        # Get Bounding box
        def point_treshold(pt, background):
            '''
            loop 2 time (width and height)
            '''
            im_size = background.size
            for i in range(2):
                if pt[i] < 0:
                    pt[i] = 0
                elif pt[i] > im_size[i]:
                    pt[i] = im_size[i]
            return tuple(pt)
            
        start_point = point_treshold([width, height], background)
        end_point   = point_treshold([width + frontImage.width, height + frontImage.height], background)

        # Print Image Verbose
        if verbose:
            plt.imshow(img_merged)
            plt.scatter(x, y)
            # plt.show()

            # DEBUG
            color = (0, 255, 0)
            thickness = 4
            cv2.rectangle(img_merged, start_point, end_point, color, thickness)
            imshow(img_merged, scale=0.8, verbose=False)
        return img_merged, [start_point, end_point, background.width, background.height]
    
            
