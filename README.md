# I Spy by Edward Ng (2023)

The classic children game I Spy, where the user chooses an object, says its colour and the opponent has to guess what that object is.

This game is written in Python and uses the OpenCV and NumPy libraries. It also uses a pretrained COCO model that is trained on the COCO dataset. This model
can detect and classify common, everyday objects.

How this game works is the user picks an object and inputs its colour in the console. The computer will then search all objects of that colour in the frame
and guess which object the user has chosen. This is done through detecting images in a HSV and masking that color to partition the frame
to only classify those objects using the COCO model. The computer will then guess among the available choices where the user will reply with yes or no depending
on if the computer guessed correctly.

A problem encountered was getting an object of the specified colour from the user and classifying it. To work around this, the colour selected is masked and
the frame of image is split based on the sizes of the bounding box around the areas where the chosen colour pops up. The model will then classify only the objects 
in these bounding boxes, appending these objects to a list of strings that the game will guess from.

In conclusion, the app is able to succesfully identify and classify objects of a selected colour with a decent amount of accuracy. Further solutions to improve this
game is by continuous learning of the current pretrained COCO model.


