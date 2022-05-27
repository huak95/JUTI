# JUTI

<div id="top"></div>
JUTI - JUst Train It (Autonomously generate bounding box labels for object detection)
<br />

<!-- ABOUT THE PROJECT -->
## About The Project
Unbalanced datasets or unlabeled data are the most common problems encountered in many object detection tasks. Furthermore, labeling little objects is a difficult task. As a result, we come up with the concept of generating labels data from unlabeled photos.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
### Colab Pre-Setup
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huak95/JUTI/blob/main/JUTI_DEMO_COLAB.ipynb)

### Local Setup
1. Clone the repo
   ```
   git clone https://github.com/huak95/JUTI.git
   ```
2. Install packages 
    ```
    pip install -r requirements.txt
    ```
<p align="right">(<a href="#top">back to top</a>)</p>

### DEMO Usage 

To use with shrine object data

1. Find background segmentation class

    ```
    # import utils 
    from utils.box_gen import *
    from utils.path_help import *
    from segment import FindLocation
    import torch

    # Start finding all class labels
    fl = FindLocation()
    all_labels = fl.get_sample_labels(0.2, "data/background/")

    # view all labels
    print("all_labels:", ", ".join(list(all_labels)))
    ```
    Results 

    > all_labels: sky, ceiling, house, earth, signboard, fence, building, road, sidewalk, water, plant, door, palm, tree, wall

2. Select background class for each object
    ```
    from segment import generate_data

    gen = generate_data(all_labels)
    sel_df = gen.get_selected_df()

    # Set backgound and object images size (in pixel)
    gen.set_background_size(500)
    gen.set_object_size(150)
    ```
<table border="1" class="dataframe" align="center">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fg</th>
      <th>bg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cat</td>
      <td>[road,earth]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dog</td>
      <td>[road]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>raccoon</td>
      <td>[road,sidewalk]</td>
    </tr>
  </tbody>
</table>

3. Inference JUTI
```
    from juti import generate_yolo_dataset_beta

    label_select = 'raccoon'

    generate_yolo_dataset_beta(gen, 
                               f"data/background/", 
                               f"data/object/{label_select}/", 
                               n_images=10, 
                               verbose=False)
```

4. Upload generated image (runs folder) to [RoboFlow](https://roboflow.com/)

<div align="center">
 <img width="400" src="https://user-images.githubusercontent.com/38836072/163729957-f3d99b5a-7a03-4176-a1f3-b9af5d22d9e1.png"></a>
</div>

3. Click Finish Uploading
4. Get labeled bounding box images Hoo Ray!
   
<div align="center">
 <img width="600" src="https://user-images.githubusercontent.com/38836072/163730009-6a24c508-c4b1-4d3f-b32a-dc99384115ac.png"></a>
</div>

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Feature-->
## Feature
- ✅   Auto-generate bounding box
- ✅   Easy to customize new datasets
- ✅   RoboFlow Support
- ✅   Accurately object locolization
- ❌   Automate multi-class images

<p align="right">(<a href="#top">back to top</a>)</p>

<!--Request Feature-->
## JUTI v 2.0 Feature
### [DETR](https://github.com/facebookresearch/detr) Image Segmentation 
To accurately locolize objects. By using pre-trained on common object to segment backgroud into diffrence mask class (floor, tree, sky, house). Then create a bounding box in a correct mask (eg. cctv object in house mask, shrine on floor mask)

<div align="center">
 <img width="300" src="https://user-images.githubusercontent.com/38836072/163730659-7c87c59e-b393-46d8-b277-a6920f734c92.png"></a>
</div>

<p align="right">(<a href="#top">back to top</a>)</p>

#### Improving procedure
1. Load background un-label data
2. Use DETR on sample image (5%) to gain all mask class
3. Load transparent image data
4. Selected image data area based on mask class 
5. Start random location of object on mask
6. Generate YOLO bounding box format
7. Happy with new DATA!

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Contact-->
## Contact
Saksorn Ruangtanusak - [Linkedin](https://www.linkedin.com/in/saksorn/)

Project Link: [https://github.com/huak95/JUTI](https://github.com/huak95/JUTI)
