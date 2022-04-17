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
### Setup
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

1. Run Programme
    ```sh
    python juti.py \
    --background_path data/background/ \
    --object_path data/object/shrine/  \
    --images_num 10 
    ```
#### Option for Argsparse

`--background_path`: path of background images

`--object_path`: path of object images

`--images_num`: number of image to generate bounding box

`--verbose`: Print when runing the functions or not

2. Upload generated image (runs folder) to [RoboFlow](https://roboflow.com/)

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

<!--Request Feature-->
## Request Feature

<!-- Contact-->
Saksorn Ruangtanusak - [Linkedin](https://www.linkedin.com/in/saksorn/)

Project Link: [https://github.com/huak95/JUTI](https://github.com/huak95/JUTI)
