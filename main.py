from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import cv2
import os

# Deklarera vart vi hittar träning och test data
trainDataset = "data/train"
testDataset = "data/test1"

# Lista alla bilder i träningstdatan
trainImagepaths = list(paths.list_images(trainDataset))

# Lista alla bilder i testdatan
testImagepaths = list(paths.list_images(testDataset))

# initialisera alla matriser och labels till träning och testdata
# Träningsdata:
rawImages = []
features = []
labels = []
# Testdata
testImages = []
testFeatures = []
testLabels = []

# Anger hur många neighbors som används för att förutsäga vad testbilden föreställer
neighbors = 5

# anger model och parametrar som ska köras med denna
model = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=-1)

# Bild till vektor konvertering.
def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

# Bild till färghistogram.
def extract_color_histogram(image, bins=(8, 8, 8)):
	# Extrahera ett 3D färghistogram från HSV color space från antal `bins` per kanal
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
	# Normalisera datan
	cv2.normalize(hist, hist)
	# Returnera flattened histogram som feature vector
	return hist.flatten()

print("[TRÄNING] dataset har {} bilder".format(len(trainImagepaths)))

# loopa alla träningsbilder
for (i, imagePath) in enumerate(trainImagepaths):
	# Ladda bild och klassifiera bilden efter vad filen är döpt till i början (cat, dog)
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	# extrahera raw pixel intensity "features", samt färghistogram
	# för att se färg distributionen på pixlarna i bilden.
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# uppdatera bild, feature, och label matriserna.
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)

	# Visa updatering var 1,000:e bild
	if i > 0 and i % 1000 == 0:
		print("[TRÄNING] bearbetat {}/{}".format(i, len(trainImagepaths)))

# dela bilderna till 80 % träning och 20% test
trainRI, testRI, trainRL, testRL = train_test_split(rawImages, labels, test_size=0.2)
trainFeat, testFeat, trainLabels, testLabels = train_test_split(features, labels, test_size=0.2)

# Träna och validera K-Nearest-Neighbor på "raw pixel intensities"
def NBCRaw():
	print("[TRÄNING] validerar raw pixel noggrannhet...")
	model.fit(trainRI, trainRL)
	acc = model.score(testRI, testRL)
	print("[TRÄNING] raw pixel noggrannhet: {:.2f}%".format(acc * 100))

NBCRaw()

# Träna och validera K-Nearest-Neighbor på "histogram features"
def NBCHistogram():
	print("[TRÄNING] validerar histogram features noggrannhet...")
	model.fit(trainFeat, trainLabels)
	acc = model.score(testFeat, testLabels)
	print("[TRÄNING] histogram features noggrannhet: {:.2f}%".format(acc * 100))

NBCHistogram()

# loopa alla testbilder
for (i, imagePath) in enumerate(testImagepaths):
	# Ladda bild och klassifiera bilden efter vad filen är döpt till i början (cat, dog)
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	# extrahera raw pixel intensity "features", samt färghistogram
	# för att se färg distributionen på pixlarna i bilden.
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# uppdatera bild, feature, och label matriserna.
	testImages.append(pixels)
	testFeatures.append(hist)
	testLabels.append(label)

	# Visa updatering var 1,000:e bild
	if i > 0 and i % 1000 == 0:
		print("[TEST] bearbetat {}/{}".format(i, len(testImagepaths)))

pred = model.predict(testFeatures)
pred = np.array([0 if x == "dog" else 1 for x in pred])

for i in range(0, len(pred)):
	result = 'hund' if pred[i] == 0 else 'katt'
	print('bild ' + str(i+1) + '.jpg är en ' + result)
