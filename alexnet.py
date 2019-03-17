# Import necessary packages
import argparse

# Import necessary components to build LeNet
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from google.colab import drive



def alexnet_model(img_shape=(224, 224, 3), n_classes=26, l2_reg=0.,
	weights=None):

	# Initialize model
	alexnet = Sequential()

	# Layer 1
	alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
		padding='same', kernel_regularizer=l2(l2_reg)))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 2
	alexnet.add(Conv2D(256, (5, 5), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(512, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 4
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))

	# Layer 5
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 6
	alexnet.add(Flatten())
	alexnet.add(Dense(3072))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 7
	alexnet.add(Dense(4096))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 8
	alexnet.add(Dense(n_classes))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('softmax'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet

# def parse_args():
# 	"""
# 	Parse command line arguments.
# 	Parameters:
# 		None
# 	Returns:
# 		parser arguments
# 	"""
# 	parser = argparse.ArgumentParser(description='AlexNet model')
# 	optional = parser._action_groups.pop()
# 	required = parser.add_argument_group('required arguments')
# 	optional.add_argument('--print_model',
# 		dest='print_model',
# 		help='Print AlexNet model',
# 		action='store_true')
# 	parser._action_groups.append(optional)
# 	return parser.parse_args()

if __name__ == "__main__":
	# Command line parameters
	#args = parse_args()
	print_model = True
	drive.mount('/content/drive')
	path = "/content/drive/My Drive/Miniproject/train"
# (1) Create Training (80%), test (20%) and validation (20%) dataset
#     Datasets (x and y) are loaded as numpy object from the previous step
#path = "/home/harsha/miniproject/ISL/datanew"
# grab all image paths in the input dataset directory, then initialize
# our list of images and corresponding class labels
	print("[INFO] loading images...")
	imagePaths = paths.list_images(path)
	data = []
	labels = []
	count = 0
	for imagePath in imagePaths:
	# load the input image from disk, resize it to 32x32 pixels, scale
	# the pixel intensities to the range [0, 1], and then update our
	# images list
		image = Image.open(imagePath)
		image = np.array(image.resize((112, 112))) / 255.0
	#cv2.imshow('image', image)
	#cv2.waitKey(1)
		count = count + 1
		print(count)
		data.append(image)

	# extract the class label from the file path and update the
	# labels list
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)

# encode the labels, converting them from strings to integers
	lb = LabelBinarizer()
	labels = lb.fit_transform(labels)
	# Create AlexNet model
	print(labels)
	model = alexnet_model()
	(trainX, testX, trainY, testY) = train_test_split(np.array(data),
		np.array(labels), test_size=0.25)
	print("[INFO] training network...")
	opt = Adam(lr=1e-3, decay=1e-3 / 50)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])
	H = model.fit(trainX, trainY, validation_data=(testX, testY),
		epochs=5, batch_size=32)
# image = Image.open('/home/harsha/miniproject/python-machine-learning/train/A/001.jpg')
# image = np.array(image.resize((32, 32))) / 255.0
# model.predict(image, batch_size=32, verbose=0, steps=None)

# evaluate the network
	print("[INFO] evaluating network...")
	predictions = model.predict(testX, batch_size=32)
	print(classification_report(testY.argmax(axis=1),
		predictions.argmax(axis=1), target_names=lb.classes_))
	model.save("hand.model")
	# Print
	if print_model:
		model.summary()