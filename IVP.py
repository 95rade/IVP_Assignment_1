import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as read
import cv2

def loadImage(name, usecv=False, type=None):
    if usecv:
        if type is not None:
            return cv2.imread(name, type)
        else:
            return cv2.imread(name)
    else:
        return read.imread(name)

def plotImage(image, usecv = False):
    if usecv:
        cv2.imshow('image', image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        plt.figure()
        plt.imshow(image)
        plt.show()

def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : list of (name, image)
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,figure in enumerate(figures):
        name, image = figure
        axeslist.ravel()[ind].imshow(image, cmap=plt.gray())
        axeslist.ravel()[ind].set_title(name)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    plt.show()

def horizontalFlip(img, isGray=False):
    height = int
    width = int
    dimension = int

    if isGray:
        height, width = img.shape
        dimension = 0
    else:
        height, width, dimension = img.shape

    for i in range(height):
        img[i] = img[i][::-1]

    return img

def verticalFlip(img):
	return img[::-1]

def flipHorizontalAndVertical():
    img = loadImage('red.jpg')

    figures = []

    figures.append(("Orignal", img))
    figures.append(("Vertical Flip", verticalFlip(img)))
    figures.append(("Horizontal Flip", horizontalFlip(img.copy())))

    plot_figures(figures, 1, 3)

def averageIntensityThresholding():
    orignalImage = loadImage('moon.tif')

    figures = []
    figures.append(("Orignal", orignalImage))

    image = orignalImage.copy()

    avgIntensity = np.sum(image) / image.size
    print("avgIntensity = ", avgIntensity)

    image[image <= avgIntensity] = 0
    image[image > avgIntensity] = 255

    figures.append(("Threshold Applied", image))

    plot_figures(figures, 1, 2)

def negativeOfImage():
    orignalImage = loadImage('moon.tif')

    figures = []
    figures.append(("Orignal Image", orignalImage))

    image = orignalImage.copy()

    image = 255 - image
    figures.append(("Negative Image", image))

    plot_figures(figures, 1, 2)

def scaleIntensityLevel(image, initialLevel, toLevel):
    return (image // (2 ** (initialLevel - toLevel))) * (2 ** (initialLevel - toLevel))

def redueIntensityLevelTo1(showWithOpenCV = False):
    image = loadImage('priyanka.tiff')

    figures = []

    for i in reversed(range(1, 9)):
        figures.append((str(i) + " bit", scaleIntensityLevel(image.copy(), 8, i)))

    plot_figures(figures, 2, 4)

def rotateImage(angle):
    image = loadImage('red.jpg', True, cv2.IMREAD_GRAYSCALE)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    linear = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    nearest = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
    cubic = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_CUBIC)
    area = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_AREA)

    figures = []

    figures.append(("INTER_LINEAR", linear))
    figures.append(("INTER_NEAREST", nearest))
    figures.append(("INTER_CUBIC", cubic))
    figures.append(("INTER_AREA", area))

    plot_figures(figures, 1, 4)

def zoom(image, factor):
    image = np.repeat(image, factor, axis=1)
    image = np.repeat(image, factor, axis=0)
    return image

def shrink(image, factor):
    return image[::factor, ::factor]

def shrinkAndZoomByFactorOf10():
    image = loadImage('red.jpg')
    factor = 10
    shrinkedImage = shrink(image.copy(), factor)
    zoomedImage = zoom(shrinkedImage.copy(), factor)

    # plotImage(shrinkedImage)

    figures = []
    figures.append(("Orignal Image", image))
    figures.append(("Shrinked Image", shrinkedImage))
    figures.append(("Zoomed Image", zoomedImage))

    plot_figures(figures, 1, 3)

def noise():
    image = loadImage('img.jpg')

    nImgaesWithNoise = []

    noisyImages = 6

    figures = []

    figures.append(("Orignal Image", image))

    averageImage = np.zeros(image.shape, dtype=int)
    for i in range(noisyImages):
        noisyImage = image.copy() + np.random.randint(50, size=image.shape)
        # empty = np.zeros(image.shape, dtype=np.uint8)
        # cv2.randn(empty, 1, 10)
        # noisyImage = image.copy() + empty
        averageImage = np.add(averageImage, noisyImage)
        figures.append(("Noisy Image", noisyImage))

    averageImage = averageImage // (noisyImages + 2)

    figures.append(("Average Image", averageImage))

    plot_figures(figures, 2, 4)

def main():
	# flipHorizontalAndVertical()			   # Q1
	# averageIntensityThresholding()		   # Q2
	# negativeOfImage()						   # Q3
	redueIntensityLevelTo1(False)            # Q4
    # shrinkAndZoomByFactorOf10()              # Q5
    # noise()                                  # Q6
	# rotateImage(45)                          # Q7


if __name__== "__main__":
  main()
