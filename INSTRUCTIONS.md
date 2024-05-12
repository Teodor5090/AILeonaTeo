# AI Project: Flower Species Classifier

Instructions: 

First you will need to install the needed packages for the app to run, you will need to write in the terminal:
NPM install tensorflow
NPM install pillow

Next, if there is no file named "flower_classifier_model.h5" in the project then the first step is to run main.py to generate a model based on which the app will later on use to predict the flower species.

After you have confirmed that flower_classifier_model.h5 does exist, open the predict.py file and on line 41 there is an image_path variable, here you will set the image path for the picture that you would like to input for a prediction.

Finally after you have completed the previous steps, run the predict.py file and you should get a prediction of what type of flower it is.

NOTE: The database we used only has 5 species (Daisy, dandelion, rose, sunflower and tulip).
