import os
import pandas as pd

def create_path(PATH, verbose=False):
    try:
        os.mkdir(PATH)
        if verbose:
            print(f"create {PATH}")
    except:
        if verbose:
            print(f"path {PATH} exist")

def get_dirs_sorted(PATH):
    all_dirs = pd.Series(os.listdir(PATH)).sort_values()
    all_dirs = PATH + all_dirs
    return all_dirs