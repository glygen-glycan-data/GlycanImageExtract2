## Create Training data
Run the following python program to create YOLO training data:  
    run build_training_data.py  

## Instructions to setup rclone for your google drive account  
create rclone config file using your systems terminal  
- install rclone on your system if not already present  
- rclone config  

Then follow the interactive setup:  
1) Choose: n (for new remote)  
2) Name it as: my_drive  
3) Storage type: Google drive  
4) Client ID/Secret: Press Enter to use default  
5) Scope: type 1 for full access to drive  
6) Root Folder ID / Service Account File: press enter  
7) Allow access to your Google Account  
8) You will use the rclone.config file created while training the model (basically upload your config file to the root directory while running the train model step)  


## Train model  
Run the training script: yolo_v3_training.ipynb
Note: Follow the instructions wrt setting up google drive folders - which is provided in yolo_v3_training.ipynb before you start training.