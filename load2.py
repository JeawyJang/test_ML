from autokeras.classifier import load_image_dataset
from autokeras.classifier import _validate
from autokeras.classifier import ImageClassifier
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import np_utils
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input


if __name__ == '__main__':


	#creating model
	model = Sequential()

	model.add(Conv2D(32, (5, 5), input_shape=(256,256,3)))
	model.add(Activation('relu'))
	BatchNormalization(axis=-1)
	model.add(Conv2D(32, (5, 5)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	BatchNormalization(axis=-1)
	model.add(Conv2D(64,(5, 5)))
	model.add(Activation('relu'))
	BatchNormalization(axis=-1)
	model.add(Conv2D(64, (5, 5)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())
	# Fully connected layer

	BatchNormalization()
	model.add(Dense(512))
	model.add(Activation('relu'))
	BatchNormalization()
	model.add(Dropout(0.2))
	model.add(Dense(4))

	# model.add(Convolution2D(10,3,3, border_mode='same'))
	# model.add(GlobalAveragePooling2D())
	model.add(Activation('softmax'))
	print(model.summary())
	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
	train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

	#Input data
	x_train, y_train = load_image_dataset(csv_file_path="training2/label.csv",
	                                      images_path="training2")
	print(x_train.shape)
	print(y_train.shape)

	x_test, y_test = load_image_dataset(csv_file_path="test/label.csv",
	                                    images_path="test")
	print(x_test.shape)
	print(y_test.shape)

	

	number_of_classes = 4

	Y_train = np_utils.to_categorical(y_train, number_of_classes)
	Y_test = np_utils.to_categorical(y_test, number_of_classes)

	print(y_train[0], Y_train[0])

	test_gen = ImageDataGenerator(rescale=1./255)
	gen = ImageDataGenerator(rotation_range=7, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)
	train_generator = gen.flow(x_train, Y_train, batch_size=160)
	test_generator = test_gen.flow(x_test, Y_test, batch_size=40)

	#testing the model
	model.fit_generator(train_generator, steps_per_epoch=640//160, epochs=6, 
                    validation_data=test_generator, validation_steps=160//40)


	#evaluate the model

	score = model.evaluate(x_test, Y_test)
	print()
	print('Test accuracy: ', score[1])
