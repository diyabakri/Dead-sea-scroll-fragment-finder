from FileReader import FileReader 
from SegPrep import SegPrep
import skimage.util as ut
import skimage.io as io
import skimage.transform as ts
import skimage.segmentation as sg
import numpy as np
import cv2
import skimage.morphology as mp
import skimage.measure as me

def main():
    

    imreader = FileReader("./results/image0","s*2.jpg")
    imList = imreader.getImages()
    prep = SegPrep(imList)

    prep.itrBilateralSmoothing(itrations=2)

    prep.changeColorSpace(colorReagons=100)
    prep.changeColorSpace(colorReagons=50)
    prep.changeColorSpace(colorReagons=25)
    prep.changeColorSpace(colorReagons=10)
    prep.changeColorSpace(colorReagons=8)
    prep.changeColorSpace(colorReagons=5)
    prep.changeColorSpace(colorReagons=4)
    # prep.morphEadgeDetection()
    # prep.cannyEadgedetection()
    prep.adaptiveThreshold()
    # prep.findBorders()
    # prep.foodFill()
    # prep.clearBoarders()
    # prep.reginalfilling()
    # prep.clearBoarders(buffer_size=220)
    # prep.label()
    # regionTable = prep.getRegionProps()
    # validregionTable = prep.getValidRegions(regionTable,minArea=10000)
    # dataFrameTable = prep.sortRegionsInImage(validregionTable)
    # prep.drawRec(dataFrameTable)
    # prep.cropRecFromPhoto(dataFrameTable)
    # prep.saveImage()


    for i in range(len(imList)):
        
        io.imsave("./results/image%d.jpg"%i,prep.prevResults[i])
    
if __name__ == "__main__":
    main()
