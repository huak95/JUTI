import argparse
from utils.box_gen import *
from utils.path_help import *
from segment import FindLocation

def parse_args():
    """
    Params:
        --background_path: str
                path of background images
                Default as "data/background"
        --object_path: str
                path of object images
                Default as "data/object/shrine"
        --images_num: int
                number of image to generate bounding box
                Default as 1
        --verbose: bool
                Print when runing the functions or not
                Default as True

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--background_path", type=str, default="data/background/")
    parser.add_argument("--object_path", type=str, default="data/object/shrine/")
    parser.add_argument("--images_num", type=int, default=1)
    parser.add_argument("--verbose", type=bool, default=True)
    return parser.parse_args()


def generate_yolo_dataset_beta(gen, BACKGROUND_PATH, OBJECT_PATH, n_images=10, verbose=True):
    '''
    Main Program of JUTI (JUst Train It)
    Create a full auto generation bounding box pipeline
    '''
    # Background
    all_background_dirs = get_dirs_sorted(BACKGROUND_PATH)
    background_dirs = all_background_dirs.sample(n_images)
    n_bg = len(background_dirs)

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

    # Loop each background to create a merged images
    img_list, loc_list = [], []
    # print(background_dirs)
    fl = FindLocation()

    seg_images, labels = fl.segment_from_list(background_dirs.to_list(), False)
    
    for idx in tqdm(range(len(background_dirs))):
        path_bg = background_dirs.values[idx]
        path_fg = all_object_dirs.sample(1).values[0]
        # print(path_bg, path_fg)
        img_merged, loc = gen.gen_box_beta(seg_images[idx], path_bg, path_fg, verbose)
        
        if type(loc) != list:
            print("Skipp")
            continue
        
        else:
            # Save to list
            img_list.append(img_merged)
            loc_list.append(loc)

    # Save File
    save_image(img_list, PATH = save_path+"/images")
    df_loc = pd.DataFrame(loc_list, columns=['point_1', 'point_2', 'width', 'height'])
    dataframe_to_yolo_f(df_loc, img_class=object_name, PATH=save_path+"/labels")
    print(f"Finally Create {object_name}")

if __name__ == "__main__":
    args = parse_args()
    generate_yolo_dataset(args.background_path,
                          args.object_path,
                          args.images_num,
                          args.verbose)
