# NTU RGBD Tubelet Parser
Please download the dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/)

**Files or Folder's of the format** 

Please ref to paper "NTU RGB+D 120: A Large-Scale Benchmark for 3D Human Activity Understanding" for more info 
```
    S<sss>C<ccc>P<ppp>R<rrr>A<aaa> 

    <sss> -> setup number , it has 32 different setups with different camera height and distance  
    <ccc> -> Camera ID  
    <ppp> -> Performer ID  
    <rrr> -> Replication number   
    <aaa> -> Activity ID  
```

**Steps to parse the tracklets**

NTU RGB consist of Zip Files with videos

1. Unzip the each file
2. Get the video file of specified activity
3. Conver the videos into frames
4. Run object detector on frames and extract the pedestrian bbox info
    1. Note - We only have one person per frame
5. Generate the tracklets
6. Save both frame and tracklets in respective destinations

## How to run ?
Step1: Create an venv and install requirements
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Step2: Download the NTURGB Dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/)

Step 3: Modify the `SRC_DIR` in `ntu_rgb_tubelet_extractor.py` to reflect the directory containing nturgb zip files 

Step 4: Run the `ntu_rgb_tubelet_extractor.py` to extract the tubelets
```
python ntu_rgb_tubelet_extractor.py
``` 