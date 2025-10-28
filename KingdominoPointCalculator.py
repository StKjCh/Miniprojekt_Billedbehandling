import cv2 
import numpy as np
import matplotlib.pyplot as plt

startImage = 1
numberOfImages = 50

###########################################################################################
#          Global variables
###########################################################################################

types = ['lake' , 'forest' , 'grass' , 'field' , 'swamp' , 'mine' , 'start' , 'none']
numCorrectData = {'lake': 0 , 'forest': 0 , 'grass': 0 , 'field': 0 , 'swamp': 0 , 'mine': 0 , 'start': 0 , 'none': 0}  # Dictionary to count correct data 
numCorrectGuess = {'lake': 0 , 'forest': 0 , 'grass': 0 , 'field': 0 , 'swamp': 0 , 'mine': 0 , 'start': 0, 'none': 0}

tileThreshold = np.array([[130,200],[115,180],[0,44],   # Thresholds Field RGB 
                          [20,48],[170,255],[135,200],  # Thresholds Field HSV
                          [43,102],[58,109],[21,56],    # Thresholds Forest with house RGB
                          [32,62],[102,171],[59,113],   # Thresholds Forest with house HSV
                          [44,93],[77,100],[50,52],     # Thresholds Forest without house RGB
                          [32,42],[103,154],[60,89],     # Thresholds Forest without house HSV
                          [0,70],[50,110],[105,190],    # Thresholds Lake with house  RGB
                          [74,110],[170,254],[123,190], # Thresholds Lake with house  HSV
                          [64,133],[87,185],[22,61],    # Thresholds Grass RGB
                          [30,50],[140,211],[94,182],    # Thresholds Grass HSV
                          [90,157],[79,140],[46,103],   # Thresholds Swamp RGB
                          [17,40],[65,151],[93,157],    # Thresholds Swamp HSV
                          [0,215],[60,190],[0,150],  # Thresholds Mine RBG
                          [18,105],[80,251],[72,211]])   # Thresholds Mine HSV

# Declare confusion matrix
confusionMatrix = np.zeros((8, 9), dtype=object)
for i, t in enumerate(types):
    confusionMatrix[i][0] = t

###########################################################################################
#          Image Initialization and Preprocessing
###########################################################################################

def LoadImage(i):
    '''
    Loading images.
    '''
    field = cv2.imread(f'Billeder/{i+1}.JPG', cv2.IMREAD_COLOR) # Read image

    field_r = field[:,:,2]
    field_b = field[:,:,0]       
    combined = field_b * 0.8 + field_r * 0.2

    ##combined_blur = cv2.GaussianBlur(combined, (3,3),0)
    combined_normed = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return combined_normed, field

def LoadTemplate():
    '''
    Loading templates for the crowns and normalize.
    '''
    template = cv2.imread(f'Billeder/Template/Krone1.jpg', cv2.IMREAD_COLOR)
    template_r = template[:,:,2]
    template_b = template[:,:,0]         
    combined = template_b * 0.8 + template_r * 0.2
    ##combined_blur = cv2.GaussianBlur(combined, (3,3),0)
    combined_normed = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return combined_normed

def LoadStartParam():
    '''
    Generate scoreboard and difine threshold for template matching for the crown.
    '''
    ##cv2.namedWindow("template", cv2.WINDOW_NORMAL)
    ##cv2.resizeWindow("template", 600,500)
    scoreBoard = np.zeros((5,5))
    threshold = 0.361
    return scoreBoard, threshold

def CropImage(img, y, x):
    '''
    Funktion to crop image to tile.
    '''
    img_crop = img[100*y:100*(y+1), 100*x:100*(x+1)]
    return img_crop


###########################################################################################
#          Image Adjustment and Normalization
###########################################################################################

def Equalize(img, temp):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    field_b_eq = clahe.apply(img)
    template_b_eq = clahe.apply(temp)
    return field_b_eq, template_b_eq

def ColorNormalize(img):
    '''
    Adjust saturation and apply CLAHE to normalize image colors.
    '''
    # Define CLASH
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    #Covert to HSV and Split channels
    h , s , v = cv2.split( cv2.cvtColor(img, cv2.COLOR_BGR2HSV) ) 
    
    # Adjust saturation and adjust the contrast with CLASH
    s_adj = clahe.apply(s)   #np.clip(s*1.2,0,255).astype(np.uint8) 
    v_clahe = clahe.apply(v) 

    # Merge channels, convert back to RGB and return normalized image
    HSVnorm = cv2.merge((h , s_adj , v_clahe)) # Megre channels
    return cv2.cvtColor(HSVnorm, cv2.COLOR_HSV2BGR)


###########################################################################################
#          Crown Detection
###########################################################################################

def TemplateMatch(img, temp, threshold):
    '''
    Template matching to find crowns.
    '''
    rotate = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    field, crown = img, temp
    xsAll, ysAll = [], []
    kernel = np.ones((3,3), np.uint8)
    field_eq, crown_eq = Equalize(field, crown)
    for i in range(4):
        if rotate[i] is not None:
            crown_rotated_eq = cv2.rotate(crown_eq, rotate[i])
        else:
            crown_rotated_eq = crown_eq
        crown_match = cv2.matchTemplate(field_eq, crown_rotated_eq, cv2.TM_CCOEFF_NORMED)
        maxima = (crown_match == cv2.dilate(crown_match, kernel)) & (crown_match >= threshold)

        ys, xs = np.where(maxima)
        if ys.size:
            xsAll.extend(xs.tolist())
            ysAll.extend(ys.tolist())
    return np.array(xsAll), np.array(ysAll)

def DrawSquare(img, x, y, w, h):
    '''
    Function to draw a squarre around detected crowns
    '''
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)


###########################################################################################
#          Tile Type Determination
###########################################################################################

def FeatureArray(img):
    '''
    Returns an array with a weighted average of R, B, G, H, S and V
    '''
    numPixel = 10000

    # Udtræk RGB-kanaler (OpenCV = BGR)
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]

    # Beregn vægtet gennemsnit for hver RGB-kanal
    histogramR = cv2.calcHist([r], [0], None, [256], [0, 256]).ravel()
    histogramG = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
    histogramB = cv2.calcHist([b], [0], None, [256], [0, 256]).ravel()

    avarageR = np.dot(histogramR, np.arange(256)) / numPixel
    avarageG = np.dot(histogramG, np.arange(256)) / numPixel
    avarageB = np.dot(histogramB, np.arange(256)) / numPixel

    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Calculate avarage of each HSV-channel
    histogramH = cv2.calcHist([h], [0], None, [256], [0, 256]).ravel()
    histogramS = cv2.calcHist([s], [0], None, [256], [0, 256]).ravel()
    histogramV = cv2.calcHist([v], [0], None, [256], [0, 256]).ravel()

    avarageH = np.dot(histogramH, np.arange(256)) / numPixel
    avarageS = np.dot(histogramS, np.arange(256)) / numPixel
    avarageV = np.dot(histogramV, np.arange(256)) / numPixel

    # Return array with [R, G, B, H, S, V]
    return np.array([avarageR, avarageG, avarageB,
                     avarageH, avarageS, avarageV])
    
def WhichType(featureArray):
    '''
    Determine the type by looking at the feature array and see it the are within thresholds for any tiletype.
    '''
    tileTypes = ["field", "forest", "forest", "lake", "grass", "swamp", "mine"]

    for i, tileGuess in enumerate(tileTypes):
        start = i * 6
        thresholds = tileThreshold[start:start+6]
        match = True

        for j in range(6):
            low, high = thresholds[j]
            value = featureArray[j]

            if not (low <= value <= high):
                match = False 
                break

        if match:
            return tileGuess

    return "none"


###########################################################################################
#          Point Calculation
###########################################################################################

def IgniteFire(x,y, scoreboard, typeboard, checked):
    '''
    Grassfire, using connected component analysis to isolate territoy. Size of territory and
    number of crowns is determent an multiplyed to get total point for it. 
    ''' 
    burnQueue = [(x,y)]             # Define burn Queue with starting point
    numCrown = scoreboard[x][y]     # Variable with number of crowns in first tile i territoy
    numTiles = 1                    # Counter for tiles 
    TerritoryType = typeboard[x][y] # Type of territory
    checked[x][y] = 1               # Mark starting tile as visited

    # Grassfire
    while burnQueue:
        currentX, currentY = burnQueue.pop()
        for xOffset , yOffset in [(0,1),(1,0),(0,-1),(-1,0)]:
            neighborX = currentX + xOffset
            neighborY = currentY + yOffset
            if 0 <= neighborX < 5 and 0 <= neighborY < 5:
                if checked[neighborX][neighborY] == 0 and typeboard[neighborX][neighborY] == TerritoryType:
                    numTiles += 1
                    numCrown += scoreboard[neighborX][neighborY]
                    burnQueue.append((neighborX,neighborY))
                    checked[neighborX][neighborY] = 1
    point = numCrown * numTiles
    print(f'{TerritoryType} with {numTiles} tiles and {numCrown} crowns. {point} points')
    return point 

def CalcScore(scoreboard, typeboard):
    '''
    Function to calculate the final score
    '''
    totalScore = 0              # Total score variable
    checked = np.zeros((5,5))   # Array to track whether a tile is visited
    
    # Locate the territories and the point for each of them and gives a total score for image
    for x in range(5):
        for y in range(5):

            # If not already visited, start a grassfire
            if checked[x][y] == 0:
                totalScore += IgniteFire(x, y, scoreboard, typeboard, checked)            
    
    print(f'Total scorer: {int(totalScore)}')
    return totalScore


###########################################################################################
#          Store and Analye Data
###########################################################################################

def SplitLine(line):
    '''
    Function to split lines form txt file by ','
    '''
    return [x.strip() for x in line.split(',')]

# Clear result txt file
with open("data/AllImagesAnalysis.txt", "w") as txt:
    txt.write("")
with open("data/AllImagesAnalysis.txt", "a") as txt:        
    line1 = f"titleThreshold:\n{tileThreshold}\n\n"
    txt.write(line1)

# Collect correct data about til types and number of crowns
with open("data/CorrectTileCrownPoint.txt", "r") as txt:
    correctDataAll = [SplitLine(line) for line in txt if line.strip() and ',' in line]

startIndex = (startImage - 1) * 25
endIndex = startIndex + numberOfImages * 25

correctData = correctDataAll[startIndex:endIndex]

#print(f"Antal tiles i subset: {len(correctData)}")


for i in range(len(correctData)):
    for j in range(len(types)): 
        if correctData[i][0] == types[j]:
            numCorrectData[types[j]] += 1
            #print(correctData[i][0]) 

#print(numCorrectData)

def IsItCorrect(a , typeGuess):
    '''
    Function to ckeck if 
    '''
    if correctData[a][0] == typeGuess:
        return 'Correct'
    else:
        return 'Wrong'

def WriteTileDataInTxt (tileType , correct , x , y , crowns , isTypeCorrect, featureValues):    
    '''
    Append line to txt file
    '''
    with open("data/AllImagesAnalysis.txt", "a") as txt:        
        line = f"Guess {tileType} is: {correct}, {x}, {y}, {crowns}, {isTypeCorrect}. R: {featureValues[0]} / G: {featureValues[1]} / B: {featureValues[2]} / H: {featureValues[3]} / S: {featureValues[4]} / V: {featureValues[5]}\n"
        txt.write(line)

def ImageScore(imgNum , totalScore , wasCorrect):
    '''
    Append line to txt file
    '''
    with open("data/AllImagesAnalysis.txt", "a") as txt:        
        line = f"In image {imgNum} the total score is {int(totalScore)}. {wasCorrect} was correct. That is {wasCorrect/25*100}%\n\n"
        txt.write(line)

def Analysis(numImages , numCorrect):
    '''
    Append total correctness to end of txt file 
    '''
    t = 0.000000000000000000001
    with open("data/AllImagesAnalysis.txt", "a") as txt:        
        line1 = f"ANALYSIS:\nFor all images {numCorrect} was correct out of {numImages*25} tiles. That is {numCorrect/(numImages*25)*100}%\n"
        txt.write(line1)
        line2 = f"Fields: {numCorrectGuess['field']} out of {numCorrectData['field']} is correct. That is {numCorrectGuess['field']/(numCorrectData['field']+t)*100}%\n"
        txt.write(line2)
        line3 = f"Forest: {numCorrectGuess['forest']} out of {numCorrectData['forest']} is correct. That is {numCorrectGuess['forest']/(numCorrectData['forest']+t)*100}%\n"
        txt.write(line3)
        line4 = f"Grass: {numCorrectGuess['grass']} out of {numCorrectData['grass']} is correct. That is {numCorrectGuess['grass']/(numCorrectData['grass']+t)*100}%\n"
        txt.write(line4)
        line5 = f"Lake: {numCorrectGuess['lake']} out of {numCorrectData['lake']} is correct. That is {numCorrectGuess['lake']/(numCorrectData['lake']+t)*100}%\n"
        txt.write(line5)
        line6 = f"Mine: {numCorrectGuess['mine']} out of {numCorrectData['mine']} is correct. That is {numCorrectGuess['mine']/(numCorrectData['mine']+t)*100}%\n"
        txt.write(line6)
        line7 = f"Start: {numCorrectGuess['start']} out of {numCorrectData['start']} is correct. That is {numCorrectGuess['start']/(numCorrectData['start']+t)*100}%\n"
        txt.write(line7)
        line8 = f"Swamp: {numCorrectGuess['swamp']} out of {numCorrectData['swamp']} is correct. That is {numCorrectGuess['swamp']/(numCorrectData['swamp']+t)*100}%\n"
        txt.write(line8)
        line9 = f"CONFUSION MATRIX\n{confusionMatrix}\n\n"
        txt.write(line9)

def CheckCrownScore(index, totalCrowns):
    difference = 0
    if True:          # Set to true if it is the training data and false if it is the test data
        checkCrowns = [9, 12, 10, 8, 9, 12, 9, 8, 11, 11, 7, 6, 11, 11, 7, 6, 10, 6, 11, 11, 10, 6, 11, 11, 11,
                       6, 9, 13, 11, 6, 9, 13, 8, 10, 11, 9, 8, 10, 11, 9, 11, 8, 11, 11, 9, 8, 11, 10, 10, 7]
    else:
        checkCrowns = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                       12, 10, 7, 7, 12, 11, 10, 9, 9, 11, 10, 9, 9, 5, 8, 18, 8, 5, 18, 8, 5, 8, 18, 8]
    difference = checkCrowns[index] - totalCrowns
    sumCrowns = sum(checkCrowns)
    # print(f'sum af Crowns: {sumCrowns}')
    return (index+1, difference, difference/checkCrowns[index]*100), sumCrowns
        #print(f'Score difference: {difference}')
    #


###########################################################################################
#          Main function
###########################################################################################

def Main():
    template = LoadTemplate()
    h, w = template.shape[0], template.shape[1]
    allPoints = []
    crownDifference = []
    totalCorrectCont = 0

    for i in range(numberOfImages):
        # Preprocessing for current image
        scoreboard, threshold = LoadStartParam()
        typeboard = [['' for _ in range(5)] for _ in range(5)]
        field, original = LoadImage(i+startImage-1)
        originalNorm = ColorNormalize(original)
        imageCrowns = 0
        imageCorrectCount = 0

        # Determine type and number of crownes in each tile in image
        for y in range(5):
            for x in range(5):

                # Crop images
                cropped = CropImage(field, y, x)
                croppedOriginalNorm = CropImage(originalNorm, y, x)

                # Determine tile type
                featureValues = FeatureArray(croppedOriginalNorm)
                tileType = WhichType(featureValues)    #whichType(croppedOriginalNorm , CT_rgb , CT_hsv)
                
                # Crown detction
                xs, ys = TemplateMatch(cropped, template, threshold)
                coords = np.column_stack((xs, ys))
                unique_coords = np.unique(coords, axis=0)
                crownNr = unique_coords.shape[0]

                # Update type and number of crowns in tile
                typeboard[y][x] = tileType 
                scoreboard[y,x] = crownNr

                # Check if it is correct
                correctness = IsItCorrect(i*25 + y * 5 + x , tileType)

                # Count number of crowns in imange
                imageCrowns = imageCrowns + crownNr

                # Count correct tiles
                if correctness == 'Correct':
                    imageCorrectCount += 1
                    totalCorrectCont += 1                   
                    if tileType in numCorrectGuess:
                        numCorrectGuess[tileType] += 1
                correctType = correctData[i*25+y*5+x][0]
                if tileType in types:
                    row = types.index(correctType) 
                    col = types.index(tileType)+1   
                    confusionMatrix[row][col] += 1

                # Data 
                WriteTileDataInTxt(tileType , correctType , x , y , crownNr , correctness , featureValues)

                # Append points
                if crownNr > 0:
                    x0 = x * 100
                    y0 = y * 100
                    for (x_rel, y_rel) in zip(xs, ys):
                        allPoints.append((x0 + int(x_rel), y0 + int(y_rel)))

        print(f'scoreboard:\n{scoreboard}')

        # Draw square around found crowns
        for (x_abs, y_abs) in allPoints:
             DrawSquare(originalNorm, x_abs, y_abs, w, h)

        # Calculate score fore current board and write in txt file
        totalScore = CalcScore(scoreboard, typeboard)
        ImageScore((i+1) , totalScore , imageCorrectCount)
        crownStats, _ = CheckCrownScore(i, imageCrowns)
        crownDifference.append(crownStats)

        # Show image
        # cv2.imshow("result", originalNorm)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print(numCorrectGuess)
        # Reset AllPoints
        allPoints = []

    # Write final tile analysis in txt file
    Analysis(numberOfImages , totalCorrectCont)

    # Crown analysis
    _, totalRealCrowns = CheckCrownScore(i, imageCrowns)
    absolute_forskelle = [entry[1] for entry in crownDifference]
    totalDifference = (sum(absolute_forskelle) / totalRealCrowns)*100
    ##print(f'total difference: {totalDifference}')
    ##print(f'Crown Difference: \n {np.array(crownDifference)}')
    
    print(imageCrowns)

    with open("data/AllImagesAnalysis.txt", "a") as f:
        f.write(f"total difference: {totalDifference} \n Crown Difference: \n {np.array(crownDifference)}")


if __name__ =="__main__":
    Main()

