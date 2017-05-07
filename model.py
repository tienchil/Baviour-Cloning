import csv
import cv2
import sklearn
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Activation, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
    	samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.3)


def generator(samples, batch_size=32):
	num_samples = len(samples)
	while True:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			correction = [0.0, 0.25, -0.25]
			for batch_sample in batch_samples:
				for i in range(3):
					name = batch_sample[i]
					image = cv2.imread(name)
					cropped = image[70:135, :]
					image = cv2.resize(cropped, (64, 64))
					images.append(image)
					images.append(cv2.flip(image, 1))
					angle = float(batch_sample[3]) + correction[i]
					angles.append(angle)
					angles.append(angle*(-1.0))

			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Network
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 64, 3), output_shape=(64, 64, 3)))
model.add(Convolution2D(24, (5, 5), activation="relu"))
model.add(Convolution2D(36, (5, 5), activation="relu"))
model.add(Convolution2D(48, (5, 5), activation="relu"))
model.add(Convolution2D(64, (3, 3)))
model.add(Dropout(0.1))
model.add(Convolution2D(64, (3, 3)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/32, 
					validation_data=validation_generator,
                    validation_steps=len(validation_samples)/32, epochs=5)
model.save("model.h5")
