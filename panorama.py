import numpy as np
import cv2
import glob
import imutils

image_paths = glob.glob('unstitchedImages/*.jpg')
images = []                    #Empty list where we are going to store all our images.


for image in image_paths:
    img = cv2.imread(image)     #Reading image.
    images.append(img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)              #We will be waiting for a key press before we continue to the next image.


imageStitcher = cv2.Stitcher_create()
error, stitched_img = imageStitcher.stitch(images)

if not error:

    cv2.imwrite("stitchedOutput.png", stitched_img)
    cv2.imshow("Stitched Img", stitched_img)
    cv2.waitKey(0)

    stitched_img = cv2.copyMakeBorder(
        stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))       #Making a border of our image

    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]      #Then we go to take a grayscale image and we are going to convert it from bgr to grayscale image and apply a threshold value so we can actually have a binary image where i either have a zero or one 
    #So we can find out where this border is from our image so we can actualy try to subtract all the black pixels away from our image  because we want to just have a clean output image where we dont have like these round corners and also black pixels in our image.

    cv2.imshow("Threshold Image", thresh_img)
    cv2.waitKey(0)

    contours = cv2.findContours(
        thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    #We are going to find contours on our binary image so we can find all these round circle corners and then we can extract or subtract later on.
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)
                #Here we are taking the contours and we are going to find the maximum area of the contour that we have found
    mask = np.zeros(thresh_img.shape, dtype="uint8")        #We are creating a mask that we can use it with out threshold image or like with our contours
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                    #We are just going to draw a rectangle around that  just because we want to crop our image so we only have like our stitched image without all the black noise around it.
    minRectangle = mask.copy()          #We are also going to have a minimum rectangle so that the minimum place where we wont get these round corners around our stitched image 
    sub = mask.copy()       #We will also have a submask here or like a soft image where we will subtract the minimum rectangle from our threshold image
    
    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)        #Here we are using morphological operations
        sub = cv2.subtract(minRectangle, thresh_img)

    contours = cv2.findContours(
        minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        #Finding contours on the minimum rectangle
                #Again we are going to find the area of interests but with our minimum rectangle instead of our threshold image
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    cv2.imshow("minRectangle Image", minRectangle)
    cv2.waitKey(0)

    x, y, w, h = cv2.boundingRect(areaOI)

    stitched_img = stitched_img[y:y + h, x:x + w]
    #Doing post processing of our image to only get the area of interest and removing all the noise that we show the we saw with the round corners and also black pixels.  
    #Then we are going to crop of image to the area of interest that we want
    cv2.imwrite("stitchedOutputProcessed.png", stitched_img)

    cv2.imshow("Stitched Image Processed", stitched_img)

    cv2.waitKey(0)


else:           #Only if the image can not be stitched together
    print("Images could not be stitched!")
    print("Likely not enough keypoints being detected!")