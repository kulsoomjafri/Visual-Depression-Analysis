#Depression Analyzer based on sentiment Analysis
Group Members: Syeda Kulsoom Jafri, Hemarshitha Adusumilli, Amr Elhussein
Details of project:
Our project will help in finding the depression in people through some factors that include their facial expression, their news feed and their twitter tweets. In addition to all these factors we are developing an app that will detect the depression factor of a person and will suggest them to take some precautionary measures such as sleep, go for a walk or meet a friend. 

###Software’s and data frames required to run the program:
```
	Python Ver - Python '3.6.6'
	Tensorflow – Tensorflow GPU v '2.1.0'
	Open cv2 version' 4.2.0'
	Matplotlib vesrion '3.2.1'
	sklearn 0.22.2.post1
	pandas - '1.0.3'
```
###Steps to run Visual sentiment Analysis:
Use folder 'Depression_analyzer.zip'
1. Install all the above libraries mentioned.
2. Import 'os' to get the current working directory for your project
3. Add the notepad++ file of 'depression Analyzer.py' in the current working directory.
4. It has all the required things to run the code.
5. Dataset is also in the folder with the name'dataset'.
6. Saved model is given in the saved_model folder
7. In order to check the response of the augmented dataset run 'augmented_depression_analyzer.py'


###Steps to run opencv:
1. on cmd line write python
2. import cv2
3. print (cv2.__file__ ) (it will tell you the location of the cv file directory)
4. copy the path and paste it in address bar.
5. In your current working directory, add a newfolder 'cascades' add the 'data' folder in it.

###Steps to predict images using opencv:
Use Folder 'face_detection_opencv.zip'
1. The folder 'babby' has all the images extracted from the live video.
2. In order to test the working of the webcam and extract images run 'capture.py'
3. press x to exit the video streaming and 'c' to save images from the video
4. To predict the test images created by Capture.py run 'face_extraction.py'
5. To load the saved images/videos load 'cap_imread.py'

