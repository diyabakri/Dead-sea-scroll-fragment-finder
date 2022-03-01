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
import pandas as pd
def main():
    imreader = FileReader("./data","*.jpg")
    imList = imreader.getImages()
    prep = SegPrep(imList)

    prep.itrBilateralSmoothing(itrations=2)
    prep.colorSpacePyramid()
    prep.morphEadgeDetection()
    prep.cannyEadgedetection()
    prep.adaptiveThreshold()
    prep.findBorders()
    prep.foodFill()
    prep.clearBoarders()
    prep.reginalfilling()
    prep.clearBoarders(buffer_size=220)
    prep.label()
    # regionTable = prep.getRegionProps()
    # validregionTable = prep.getValidRegions(regionTable,minArea=10000)
    # dataFrameTable = prep.sortRegionsInImage(validregionTable)
    # prep.drawRec(dataFrameTable)
    # prep.cropRecFromPhoto(dataFrameTable)


    for i in range(len(imList)):
        
        io.imsave("./results/image%d.jpg"%i,prep.prevResults[i])
    
if __name__ == "__main__":
    main()
