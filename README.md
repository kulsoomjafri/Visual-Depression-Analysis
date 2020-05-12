# Depression Analyzer based on sentiment Analysis
Group Members: Syeda Kulsoom Jafri
Details of project:
This depression Analyzer is based on three classes, Happy, Depressed and Neutral. This will predict the emotions behind an images given to it. 
Dataset which is used to train the model is from kaggle.

### Software’s and data frames required to run the program:
```
	Python Ver - Python '3.6.6'
	Tensorflow – Tensorflow GPU v '2.1.0'
	Open cv2 version' 4.2.0'
	Matplotlib vesrion '3.2.1'
	sklearn 0.22.2.post1
	pandas - '1.0.3'
```
### Steps to run Depression Analyzer:
Use Depression_analyzer.py'
1. Install all the above libraries mentioned.
2. Import 'os' to get the current working directory for your project
3. Add the notepad++ file of 'depression Analyzer.py' in the current working directory.
4. It has all the required libraries and dataframes to run the code.
5. Dataset downloaded from Kaggle is also in the folder 'dataset'.
6. Saved model is given in the saved_model folder
7. In order to check the response of the augmented dataset run 'augmented_depression_analyzer.py'

### Steps for running on an Embedded environment
1. Use the tflite file to deploy this model on an Embedded environment eg. Raspberry pi
2. Intsall Tflite interpreter from tensorflow documentation which is provided here "https://www.tensorflow.org/lite/guide/python"
