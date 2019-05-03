import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import PossiblePlate

import pytesseract
from PIL import Image
import time

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
config = ('-l eng --oem 3 --psm 6')

showSteps = False

###################################################################################################
def russia(result1, result2, result3):
    result1.replace(" ", "")
    output = result1
    valid = True
    j = 0
    
    if len(result1) > 9 or len(result1) < 8:
        valid = False
        
    for i in result1:
        if j == 0 or j == 4 or j == 5:
            if not ((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                valid = False
                break
        if j == 1 or j == 2 or j == 3 or j == 6 or j == 7 or j == 8:
            if not (ord(i) >= 48 and ord(i) <= 57):
                valid = False
                break
        if j == 8:
            if not ((ord(i) >= 48 and ord(i) <= 57) or (ord(i) <= 32 and ord(i) >= 0)):
                valid = False
                break
        j += 1

    if valid:

        return output

    else:

        valid = True
        j = 0
        output = result2

        if len(result2) != 8 or len(result2) != 9:
            valid = False

        for i in result2:
            if j == 0 or j == 4 or j == 5:
                if not ((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                    valid = False
                    break
            if j == 1 or j == 2 or j == 3 or j == 6 or j == 7 or j == 8:
                if not (ord(i) >= 48 and ord(i) <= 57):
                    valid = False
                    break
            j += 1

        if valid:

            return output

        else:

            valid = True
            j = 0
            output = result3

            if len(result3) != 8 or len(result3) != 9:
                valid = False

            for i in result3:
                if j == 0 or j == 4 or j == 5:
                    if not ((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                        valid = False
                        break
                if j == 1 or j == 2 or j == 3 or j == 6 or j == 7 or j == 8:
                    if not (ord(i) >= 48 and ord(i) <= 57):
                        valid = False
                        break
                j += 1

            if valid:

                return output

            else:
                
                valid = True
                j = 0
                output = ""
    
                for i in result2:
                    if result3.find(i) != -1:
                        j += 1
                    if j == 1:
                        output += i
                        j = 0

                if len(output) != 8 or len(output) != 9:
                    valid = False
                    
                j = 0
                for i in output:
                    if j == 0 or j == 4 or j == 5:
                        if not ((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                            valid = False
                            break
                    if j == 1 or j == 2 or j == 3 or j == 6 or j == 7 or j == 8:
                        if not (ord(i) >= 48 and ord(i) <= 57):
                            valid = False
                            break
                    j += 1

                if valid:

                    return output

                else:

                    return ""

def kazakhstan(result1, result2, result3):

    output = result1
    valid = True
    j = 0

    if len(result1) != 7:
        valid = False
        
    for i in result1:
        if j == 0 or j == 4 or j == 5 or j == 6:
            if not ((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                valid = False
                break
        if j == 1 or j == 2 or j == 3:
            if not (ord(i) >= 48 and ord(i) <= 57):
                valid = False
                break
        j += 1
    
    if valid:

        return output

    else:

        output = result1
        valid = True
        j = 0

        if len(result1) != 8:
            valid = False
            
        for i in output:
            if j == 0 or j == 1 or j == 2 or j == 6:
                if not (ord(i) >= 48 and ord(i) <= 57):
                    valid = False
                    break
            if j == 3 or j == 4 or j == 5:
                if not((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                    valid = False 
                    break 
            if j == 7:
                if not ((ord(i) >= 48 and ord(i) <= 57) or (ord(i) <= 32 and ord(i) >= 0)):
                    valid = False
                    break 
            j += 1

        if valid:
        
            return output
        
        else:

            valid = True
            j = 0
            output = result2

            if len(output) != 8:
                valid = False
                
            for i in output:
                if j == 0 or j == 1 or j == 2 or j == 6 or j == 7:
                    if not (ord(i) >= 48 and ord(i) <= 57):
                        valid = False
                        break
                if j == 3 or j == 4 or j == 5:
                    if not((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                        valid = False 
                        break  
                j += 1

            if valid:

                return output

            else:

                valid = True
                j = 0
                output = result3

                if len(output) != 8:
                    valid = False
                    
                for i in output:
                    if j == 0 or j == 1 or j == 2 or j == 6 or j == 7:
                        if not (ord(i) >= 48 and ord(i) <= 57):
                            valid = False
                            break
                    if j == 3 or j == 4 or j == 5:
                        if not ((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                            valid = False   
                            break
                    j += 1

                if valid:

                    return output

                else:
                    
                    valid = True
                    j = 0
                    output = ""
        
                    for i in result2:
                        if result3.find(i) != -1:
                            j += 1
                        if j == 1:
                            output += i
                            j = 0

                    if len(output) != 8:
                        valid = False
                        
                    j = 0
                    for i in output:
                        if j == 0 or j == 1 or j == 2 or j == 6 or j == 7:
                            if not (ord(i) >= 48 and ord(i) <= 57):
                                valid = False
                                break
                        if j == 3 or j == 4 or j == 5:
                            if not ((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                                valid = False  
                                break 
                        j += 1

                    if valid:

                        return output

                    else:

                        return ""

def main(frame):

    plateCoordinates = []

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print("\nERROR: KNN traning was not successful!\n")  # show error message
        return "", []                                          # and exit program
    # end if

    imgOriginalScene = frame

    if imgOriginalScene is None:                            # if image was not read successfully
        print("\nERROR: image not read from file!\n\n")  # print error message to std out
        return "", []                                             # and exit program
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        #print("\nNO license plates were detected!\n")  # inform user no plates were found
        return "", []
    else:
        #cv2.imwrite("detected.jpg", frame)                                                       # else
                # if we get in here list of possible plates has at leat one plate

                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

#----------------------------------------------------------------------------------------------

        img = cv2.cvtColor(licPlate.imgPlate, cv2.COLOR_BGR2GRAY)

        # Apply dilation and erosion to remove some noise
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations = 1)
        img = cv2.erode(img, kernel, iterations = 1)

        result = pytesseract.image_to_string(img)

        result.replace(" ", "")

        filtered_result = ""
    
        for i in result:
                if (ord(i) >= 48 and ord(i) <= 57) or (ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122):
                        filtered_result += i

        j = 0

        for i in filtered_result:
            if j == 0 or j == 1 or j == 2 or j == 3:
                if (ord(i) >= 48 and ord(i) <= 57):
                    j += 1
        
        if j == 4:
            filtered_result = filtered_result[1:]

        j = 0
        
        for i in filtered_result:
            if j == 4 or j == 5 or j == 6 or j == 7:
                if (ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122):
                    j += 1

        if j == 4:
            filtered_result = filtered_result[:6] + filtered_result[7:]

#----------------------------------------------------------------------------------------------

        enlargedImg3 = cv2.resize(licPlate.imgPlate, (0, 0), fy = 1.6, fx = 1.6)
        tesseractEnlarged3 = str(pytesseract.image_to_string(enlargedImg3, config = config))
        tesseractEnlarged3.replace(" ", "")

        if len(tesseractEnlarged3) > 8:
            tesseractEnlarged3 = tesseractEnlarged3[1:]
            tesseractEnlarged3 = tesseractEnlarged3[:6] + tesseractEnlarged3[7:]

        enlargedImg4 = cv2.resize(licPlate.imgPlate, (0, 0), fy = 2, fx = 2)
        tesseractEnlarged4 = str(pytesseract.image_to_string(enlargedImg4, config = config))
        tesseractEnlarged4.replace(" ", "")

        if len(tesseractEnlarged4) > 8:
            tesseractEnlarged4 = tesseractEnlarged4[1:]
            tesseractEnlarged4 = tesseractEnlarged4[:6] + tesseractEnlarged4[7:]
            tesseractEnlarged4 = tesseractEnlarged4[:8]

        plateCoordinates = drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate

        licPlate.strChars = russia(filtered_result, tesseractEnlarged3, tesseractEnlarged4)

        if licPlate.strChars == "":
            licPlate.strChars = kazakhstan(filtered_result, tesseractEnlarged3, tesseractEnlarged4)

        licPlate.strChars = licPlate.strChars.upper()

        if len(licPlate.strChars) == 0:  
            print("\nNo characters were detected!\n\n") 
            return "", []

    return licPlate.strChars, plateCoordinates
# end main

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

    return p2fRectPoints
# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0                    # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))           # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

            # unpack roatated rect into center point, width and height, and angle
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # write the chars in below the plate
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize                # unpack text size width and height

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          # based on the text area center, width, and height

            # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function


















