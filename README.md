# I Spy by Edward Ng (2023)

User picks an object and inputs its colour in the console. The computer will then search all objects of that colour in the frame
and guess which object the user has chosen. This is done through detecting images in a HSV and masking that color to partition the frame
to only classify those objects using the COCO model. The computer will then guess among the available choices.

> Uses OpenCV and NumPy

> Uses COCO dataset and model

