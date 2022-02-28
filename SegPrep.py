from array import ArrayType
import re
from skimage.io import imread 
from skimage import exposure
from skimage.transform import resize
import skimage.filters.rank as rank
from skimage.morphology import disk , binary
import skimage.morphology as mp
from skimage.feature import hog,canny
import skimage.segmentation as sg
import numpy as np
import cv2
import skimage.measure as me
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import skimage.io as io
from skimage.color import rgb2gray

class SegPrep:

    orignalImages:ArrayType
    smoothedImages:ArrayType
    hogImages:ArrayType
    binaryImages:ArrayType
    qountImages:ArrayType
    morphedImages:ArrayType
    cannyImages:ArrayType
    borderImgaes:ArrayType
    filledImgaes:ArrayType
    clearBorderImgaes:ArrayType
    prevResults:ArrayType
    reginlaFillingImgaes:ArrayType
    

    def __init__(self,orignalImages:ArrayType):
        self.orignalImages = orignalImages
        self.prevResults = orignalImages.copy()
        self.colored_image = orignalImages.copy()
        self.dict = {}
        self.df = pd.DataFrame(data=self.dict)
        self.coord_df = pd.DataFrame(data=self.dict)
        self.slices= []
    
    def reginalfilling(self,imageList=None):
        if(imageList == None):
            imageList = self.prevResults.copy()
        self.reginlaFillingImgaes = imageList.copy()
        self.foodFill(imageList = imageList.copy())

        for i in range(len(self.reginlaFillingImgaes)):
            invrimage = np.where(self.prevResults[i]>0,0,255)
            self.reginlaFillingImgaes[i] = np.add(invrimage,self.reginlaFillingImgaes[i])

        self.prevResults = self.reginlaFillingImgaes.copy()
        return self.prevResults.copy()
             
    def itrBilateralSmoothing(self,imageList=None ,footprint=disk(60),s0=150,s1=150,itrations = 1):
        if(imageList == None):
            imageList = self.prevResults.copy() 
        self.smoothedImages = imageList.copy()
        for i in range(len(self.smoothedImages)):            
            for j in range(itrations):
                self.smoothedImages[i] = cv2.blur(self.smoothedImages[i],(5,5),cv2.CV_32FC2)
                # self.smoothedImages[i] = rank.mean_bilateral(self.smoothedImages[i],selem = selem , s0=s0 , s1 = s1)
        self.prevResults = self.smoothedImages.copy()
        return self.prevResults.copy()

    def hogTrans(self,imageList = None ,orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=True):
        if(imageList == None):
            imageList = self.prevResults.copy()
        self.hogImages = imageList.copy()
        for i in range(len(imageList)):
            self.hogImages[i] = hog(self.hogImages[i],orientations=orientations,pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,visualize=visualize)
        self.prevResults = self.hogImages.copy()
        return self.prevResults.copy()

    def adaptiveThreshold(self,imageList = None,block_size = 35, method='gaussian', offset=10, mode='reflect', param=None, cval=0):
        if(imageList == None):
            imageList = self.prevResults.copy()
        self.binaryImages = imageList.copy()
        for i in range(len(self.binaryImages)):
            self.binaryImages[i] = cv2.adaptiveThreshold(self.binaryImages[i],255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,1)
            self.binaryImages[i].astype(np.uint8)
        self.prevResults = self.binaryImages.copy()
        return self.prevResults.copy()

    def changeColorSpace(self,imageList = None , colorReagons = 8):

        if(imageList == None):
            imageList = self.prevResults.copy()

        self.qountImages = imageList.copy()
        
        for i in range(len(imageList)):
            max = imageList[i].max()
            min = imageList[i].min()
            d = max - min
            regonsOffset = (int)(d/colorReagons)
            for j in range(colorReagons+1):
                imageList[i] = imageList[i].astype(np.uint8)
                currVal = regonsOffset*j
                imageList[i] = np.where( ((imageList[i]>currVal) & (imageList[i]<=currVal+regonsOffset)),currVal,imageList[i] )
        self.qountImages = imageList
        self.prevResults = self.qountImages.copy()
        return self.prevResults.copy()

    def morphEadgeDetection(self,imageList = None,openingKernal=None,closingKernal=None,dilatKernal=None,erodeKernal=None):

        if(imageList == None):
            imageList = self.prevResults.copy()
        self.morphedImages = imageList.copy()

        if(closingKernal == None):
            closingKernal = disk(5)
        if(openingKernal == None):
            openingKernal = disk(3)
        if(erodeKernal == None):
            erodeKernal = disk(3)
        if(dilatKernal == None):
            dilatKernal = disk(3)
        for i in range (len(self.morphedImages)):
            # closing = self.morphedImages[i]
            
            # close = mp.closing(self.morphedImages[i],footprint=closingKernal)
            erode = mp.erosion(self.morphedImages[i],footprint=  erodeKernal).astype(np.uint8)
            erode = mp.erosion(erode,footprint= erodeKernal).astype(np.uint8)
            dilate = mp.erosion(self.morphedImages[i],footprint= dilatKernal,).astype(np.uint8)
            edge = dilate - erode

            # edge = mp.opening(edge).astype(np.uint8)
            self.morphedImages[i] = edge
        self.prevResults = self.morphedImages.copy()
        return self.prevResults.copy()

    def cannyEadgedetection(self,imageList = None,sigma = 1):
        if(imageList == None):
            imageList = self.prevResults.copy()
        self.cannyImages = imageList.copy()
        for i in range(len(self.cannyImages)):
            self.cannyImages[i] = canny(imageList[i],sigma=sigma).astype(np.uint8)
            self.cannyImages[i]*=255
        self.prevResults = self.cannyImages.copy()
        return self.prevResults.copy()
        
    def findBorders(self,imageList = None , connectivity=1, mode='thick', background=0):
        if(imageList == None):
            imageList = self.prevResults.copy()
        self.borderImgaes = imageList.copy()
        for i in range(len(self.borderImgaes)):
            self.borderImgaes[i] = sg.find_boundaries(self.borderImgaes[i],connectivity=connectivity,mode=mode,background=background).astype(np.uint8)*255
        self.prevResults = self.borderImgaes.copy()
        return self.prevResults.copy()
                
    def foodFill(self, imageList = None , seed_point=(0,0) , new_value=255 , footprint=None, connectivity=None, tolerance=None, in_place=False, inplace=None):
        if(imageList == None):
            imageList = self.prevResults.copy()
        self.filledImgaes = imageList.copy()
        for i in range(len(self.filledImgaes)):
            if seed_point[0] < 0 or seed_point[1] < 0:
                new_seed =np.array(self.filledImgaes[i].shape)
                new_seed -= 1

                self.filledImgaes[i] = sg.flood_fill(self.filledImgaes[i],tuple(new_seed),new_value)
            else:    
                self.filledImgaes[i] = sg.flood_fill(self.filledImgaes[i],seed_point,new_value)
        self.prevResults = self.filledImgaes.copy()
        return self.prevResults.copy()
        
    def clearBoarders(self , imageList = None ,  buffer_size=0, bgval=0, in_place=False, mask=None):
        if(imageList == None):
            imageList = self.prevResults.copy()
        self.clearBorderImgaes = imageList.copy()
        for i in range(len(self.clearBorderImgaes)):
            self.clearBorderImgaes[i] = sg.clear_border(self.clearBorderImgaes[i],buffer_size=buffer_size,bgval=bgval,in_place=in_place,mask=mask)
        self.prevResults = self.clearBorderImgaes.copy()
        return self.prevResults.copy()

    def erode(self , imageList = None ,disk_size = 4):
        if (imageList == None):
            imageList = self.prevResults.copy()
        for i in range(len(imageList)):
            self.prevResults[i] = mp.erosion(imageList[i], footprint=disk(disk_size)).astype(np.uint8)
        return self.prevResults.copy()

    def open(self, imageList=None, disk_size=4):
        if (imageList == None):
            imageList = self.prevResults.copy()

        for i in range(len(imageList)):
            self.prevResults[i] = mp.opening(imageList[i], footprint=disk(disk_size)).astype(np.uint8)
        return self.prevResults.copy()

    def close(self, imageList=None, disk_size=4):
        if (imageList == None):
            imageList = self.prevResults.copy()

        for i in range(len(imageList)):
            self.prevResults[i] = mp.closing(imageList[i], footprint=disk(disk_size)).astype(np.uint8)
        return self.prevResults.copy()

    def whiteTopHat(self, imageList=None, disk_size=4):
        if (imageList == None):
            imageList = self.prevResults.copy()

        for i in range(len(imageList)):
            self.prevResults[i] = mp.white_tophat(imageList[i], footprint=disk(disk_size)).astype(np.uint8)
        return self.prevResults.copy()

    def label(self, imageList=None):
        if (imageList == None):
            imageList = self.prevResults.copy()

        for i in range(len(imageList)):
            self.prevResults[i] = me.label(imageList[i])
        return self.prevResults.copy()

    def sortRegionsInImage(self , regionBoxTable):

        dataFrames = []

        for i in range(len(regionBoxTable)):

            dict = {'up left X': [], 'up left Y': [], 'down right X': [], 'down right Y': [],"distance": []}
            df = pd.DataFrame(data=dict)
            
            for index, box in enumerate(regionBoxTable[i]):
            
                    minr, minc, maxr, maxc = box
            
                    distance_from_zero = ((minr) ** 2 + (minc) ** 2) ** (1 / 2)
            
                    df.loc[-1] = [minr , minc , maxr , maxc ,int(distance_from_zero)]  # adding a row
            
                    df.index = df.index + 1  # shifting index
            
                    df = df.sort_index()
            
            df = df.sort_values("distance")
            df.insert(0, 'ID', range(1, 1 + len(df)))
            dataFrames.append(df)

        
        return dataFrames

    def getRegionProps(self,imageList = None):
        
        if (imageList == None):
            imageList = self.prevResults.copy()
        regionsTable = []

        for i in range(len(imageList)):
            region = me.regionprops(imageList[i])
            regionsTable.append(region)
        
        return regionsTable
        
    def getValidRegions(self, regionTable , minArea = 15000):

        validRegionTable = []
        
        for i in range(len(regionTable)):

            validRegions = []

            for _,region in enumerate(regionTable[i]):
            
                if region.area >= minArea:

                    minr, minc, maxr, maxc = region.bbox
            
                    if ((maxc - minc) / (maxr - minr)) > 5 or (maxr - minr) /(maxc - minc) > 4.5:
                        continue
                    
                    validRegions.append([minr, minc, maxr, maxc])
            
            validRegionTable.append(validRegions)
        
        return validRegionTable

    def drawRec(self,  sortedBoxTable, imageList=None ):
        

        for i,table in enumerate(sortedBoxTable):

            fig,ax = plt.subplots(figsize = (10,6))
            ax.imshow(self.orignalImages[i])

            for j in range(len(table)): 
                
                currRow =table.loc[table["ID"]==j+1].iloc[0]

                minc = currRow["up left Y"]
                minr = currRow["up left X"]
                maxc = currRow["down right Y"]
                maxr = currRow["down right X"]

                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)

                row_center = minr + (maxr - minr) // 2
                column_center = minc + (maxc - minc) // 2
                ax.annotate(j+1, (column_center , row_center), color='w', weight='bold',fontsize=6, ha='center', va='center')

            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig("./results/image%i/image%i.jpg"%(i,i))

    def saveImageNoRectangle(self ,imageList=None  ,part_name=''):
        
        if (imageList == None):
            imageList = self.prevResults.copy()

        for i in range(len(imageList)):
            io.imsave("./results/image%d/image%d%s.jpg" % (i  ,i,part_name), self.prevResults[i])

    def fullSizeImageWithTheSlice(self ,i,serial , minr , minc , maxr , maxc ):

        new_image =np.zeros(self.orignalImages[i].shape)

        for row in range(minr, maxr):

            for col in range(minc , maxc):

                new_image[row , col] = self.orignalImages[i][row,col]

        io.imsave("./slice/image%d/image%d%s.jpg" % (i, i,serial), new_image)

    def cropRecFromPhoto(self,dataFrameTable,imageList = None):
        if (imageList == None):
            imageList = self.orignalImages.copy()
        for i in range(len(imageList)):
            currImage = imageList[i]
            currDataFrame = dataFrameTable[i]
            
            
            for j in range(len(currDataFrame)): 
                
                row =currDataFrame.loc[currDataFrame["ID"]==j+1].iloc[0]


                minc = row["up left Y"]
                minr = row["up left X"]
                maxc = row["down right Y"]
                maxr = row["down right X"]
                
                cropSlice = np.zeros(currImage.shape)
                
                for r in range(minr,maxr):
                    for c in range(minc,maxc):
                        cropSlice[r,c] = currImage[r,c]
                
                io.imsave("./results/image%d/slice%d.jpg"%(i,j+1),cropSlice)

    def colorSpacePyramid(self,imageList = None , pyramid = [100,50,25,10,8,5,4]):
        if(imageList == None):
            imageList = self.prevResults.copy()
        for i in range(len(imageList)):
            self.changeColorSpace(imageList,pyramid[i])
        return self.prevResults.copy()