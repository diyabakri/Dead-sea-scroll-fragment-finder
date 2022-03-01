from FileReader import FileReader
from SegPrep import SegPrep
import skimage.io as io


def main():
    imreader = FileReader("./data", "*.jpg")
    imList = imreader.getImages()
    prep = SegPrep(imList)

    prep.itrSmoothing(itrations=2)
    prep.colorSpacePyramid()
    prep.morphEadgeDetection()
    prep.findBorders()
    prep.floodFill()
    prep.clearBoarders()
    prep.reginalfilling()
    prep.clearBoarders(buffer_size=220)
    prep.label()
    regionTable = prep.getRegionProps()
    validregionTable = prep.getValidRegions(regionTable, minArea=10000)
    dataFrameTable = prep.sortRegionsInImage(validregionTable)
    prep.drawRec(dataFrameTable)

    for i in range(len(imList)):
        io.imsave("./results/image%d.jpg" % i, prep.prevResults[i])


if __name__ == "__main__":
    main()