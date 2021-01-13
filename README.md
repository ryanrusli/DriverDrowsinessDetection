# DriverDrowsinessDetection  

In the project folder, create the following directories:  

data/training/  
data/training/Alert/  
data/training/Drowsy/  
  
data/testing/  
data/testing/Alert/  
data/testing/Drowsy/  
  
data/sequences/  
data/sequences/training/  
data/sequences/testing/  
  
### extract_frames.py  
To generate frames  
Usage:  
python extract_frames.py [Input_Video_Path] [Output_Path] [Number_of_frames]  

All of the frames should be stored in either the "data/testing/" or "data/training/" folders  
An example of the output path would be data/training/Alert/[Video_Name]  
  
### extract_feautures.py  
Used to extract the facial features in the image  
Create a file called "create_sequences.csv"  
  
In the file, name the videos you want to generate sequences for in following format:  
[testing/training],[Alert/Drowsy],[Video_Name]  
  
### trainModel.py  
Run this file to train the model with the sequence you have in the "data/sequences/training" and "data/sequences/testing" folders  
