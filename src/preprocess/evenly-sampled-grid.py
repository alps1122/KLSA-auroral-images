from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import src.util.paseLabeledFile as plf


def isWithinCircle(grid, centers, radius):
    flag = True
    [h, w] = [grid[2], grid[3]]
    upperLeft = np.array([grid[0], grid[1]])
    upperRight = np.array([grid[0], grid[1] + w - 1])
    lowerLeft = np.array([grid[0] + h - 1, grid[1]])
    lowerRight = np.array([grid[0] + h - 1, grid[1] + w - 1])
    coordinates = [upperLeft, upperRight, lowerLeft, lowerRight]
    for c in coordinates:
        if np.linalg.norm(c - centers) > radius:
            flag = False
            break
    return flag


def generateGrid(imageSize, gridSize, sizeRange=(10, 30)):
    """According to image size and grid size, this function generates a evenly grid
    that coded by upper-left coordinates and its with and height, finally returns these
    grids selected within the inscribed circle where aurora occurs.

    input: imagSize [H, W], gride size [h1 w1], sizeRange
    output: gridList"""
    [w_num, h_num] = np.floor(imageSize / gridSize)
    w_num = int(w_num)
    h_num = int(h_num)
    x_map = np.tile(np.array(range(w_num)), [h_num, 1]) * gridSize[0]
    y_map = np.tile(np.array(range(h_num)).reshape([h_num, 1]), [1, w_num]) * gridSize[1]
    w_map = np.random.randint(sizeRange[0], sizeRange[1] + 1, size=(h_num, w_num))
    h_map = np.random.randint(sizeRange[0], sizeRange[1] + 1, size=(h_num, w_num))

    gridList = []
    centers = np.array([(float(imageSize[0]) - 1) / 2, (float(imageSize[1]) - 1) / 2])  # index from 0
    radius = imageSize[0] / 2
    for i in range(h_num):
        for j in range(w_num):
            grid = (x_map[i, j], y_map[i, j], w_map[i, j], h_map[i, j])
            if isWithinCircle(grid, centers, radius):
                gridList.append(grid)
    return gridList


def generateGridPatchData(imgFile, gridSize, sizeRange):
    im = Image.open(imgFile)
    im = np.array(im)
    imageSize = np.array(im.shape)
    gridList = generateGrid(imageSize, gridSize, sizeRange)

    gridPatchData = []
    for grid in gridList:
        if im.ndim == 2:
            patch = im[grid[0]:(grid[0]+grid[2]), grid[1]:(grid[1]+grid[3])].copy() # grid format: [x, y, h, w]
        if im.ndim == 3:
            patch = im[grid[0]:(grid[0] + grid[2]), grid[1]:(grid[1] + grid[3]), :].copy()  # grid format: [x, y, h, w]
        gridPatchData.append(patch)
    return gridPatchData, gridList, im

if __name__ == '__main__':

    labelFile = '../../Data/balanceSampleFrom_one_in_minute.txt'
    imagesFolder = '../../Data/labeled2003_38044/'
    imgType = '.bmp'
    gridSize = np.array([10, 10])
    sizeRange = (10, 30)

    [images, labels] = plf.parseNL(labelFile)

    imgFile = imagesFolder + images[0] + imgType

    # im = Image.open(imgFile)
    # im = np.array(im)

    # imageSize = np.array(im.shape)
    # gridList = generateGrid(imageSize, gridSize, sizeRange)
    gridPatchData, gridList, im = generateGridPatchData(imgFile, gridSize, sizeRange)

    plf.showGrid(im, gridList)
    plt.show()
