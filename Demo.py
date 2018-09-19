import keras
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import shutil
import math
import cv2
import os

# Transfer learning VGG imagnet(1000classes) parameters & customize model for few classes e.g.(dog, cat)
# by using loaded previous weight & fintune custom model to your own weight


# some basic configurations
batch_size = 32  # for train & val. modify according to your GPU memory and data
num_classes = 2  # total classes to train
base_lr = 1e-3  # learning rate
num_epoch = 30  # go through your training & val data epoch times
img_height = img_width = 224  # picture size (imagenet challenge)
channel = 3  # RGB=3, grayscale=1
seed = random.randint(0, 100)  # use for shuffle=True (not recommend, shuffle will be bad for debugging)
test_batch_size = 10  # for test
# TODO : Add your own  full path to ModelCheckpoint (checkpoint_path)
checkpoint_path = 'D:/s1040328/input_pipeline/checkpoints/weights.{epoch:02d}-{loss:.3f}-{acc:.3f}-{val_loss:.3f}-{val_acc:.3f}.hdf5'  # path for saving checkpoints


# test_data generator
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'data/test/',
    target_size=(img_width, img_height),
    color_mode='rgb',
    batch_size=test_batch_size,
    class_mode=None,
    shuffle=False,
    seed=seed)


#######################################################################################################################

# showing all images by a batch
# only the generator with shuffle=False works with .reset() so that you can get the same images in the same batch
# run test_generator.reset() if you tested the code in this block
img_list = []
for batch in test_generator:
    for i in range(test_batch_size):
        img_list.append(batch[i])
    break

for pic in img_list:
    pic = cv2.resize(pic, (224, 224), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('test', pic)
    cv2.waitKey()

cv2.destroyAllWindows()
test_generator.reset()

#######################################################################################################################

# dictionary for augmentation
data_gen_args = dict(rescale=1./255,
                     rotation_range=5,
                     shear_range=0.2,
                     zoom_range=0.2,
                     vertical_flip=True,
                     horizontal_flip=True)

# train data generator
train_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

train_generator = train_datagen.flow_from_directory(
        'data/train/',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True)


# val_data generator
val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_directory(
        'data/val/',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True)


model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, channel))
x = Flatten()(model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax')(x)
custom_model = Model(inputs=model.input, outputs=x)
custom_model.summary()
# custom_model.load_weights('weights.15-0.042-0.988.hdf5', by_name=True)  # only used when interface and fintune


# Configures the learning process
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
    return lrate


callbacks = [keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True),
             keras.callbacks.LearningRateScheduler(step_decay)]


# optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9)
# optim = keras.optimizers.Adam(lr=base_lr)
# optim = keras.optimizers.RMSprop(lr=base_lr, decay=0.9)
optim = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
custom_model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])


STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size
# training with fit_generator
# you may stop training if the model achieved val_accuracy=98.XXX% & val_loss=0.0XXX (cat & dog)
history = custom_model.fit_generator(generator=train_generator,
                                     steps_per_epoch=STEP_SIZE_TRAIN,
                                     validation_data=validation_generator,
                                     validation_steps=STEP_SIZE_VALID,
                                     epochs=num_epoch,
                                     callbacks=callbacks,
                                     initial_epoch=0,
                                     verbose=1,
                                     workers=1)


# plot model loss & save
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('VGG16_custom_model_loss_summary_graph.png')
plt.show()


# plot model accuracy & save
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('VGG16_custom_model_accuracy_summary_graph.png')
plt.show()


########################################################################################################################

# evaluate model
val_scores = custom_model.evaluate_generator(generator=validation_generator, verbose=1, max_queue_size=50, workers=1)
print('\n[ValidationData] evaluate result\nAccuracy: {0:.3f}%\nLoss: {1:.3f}'.format(val_scores[1] * 100, val_scores[0]))

########################################################################################################################

# (method-1) testing all the data in the test directory (decompose batch to per image & predict)

test_generator.reset()   # get batch_index to 0
# testing images
# test_generator.reset()  # gets to the most previous data ==> i=0
# make the test_generator.n % test_generator.batch_size = 0
for i in range(1, test_generator.n//test_generator.batch_size + 1):   # iter data by batches
    # every time you called .next() it will return you a batch of data
    # temp : ndarray [[batch, img_width, img_height, channel], X]] (if class_mode=None there is no label to return in X)
    temp = test_generator.next()
    for j in range(test_generator.batch_size):
        image = temp[j]   # get a image in the batch (img_width, img_height, channel)
        # get the value of last output Dense layer : 2 nodes in this case(cat & dog)
        # [prediction for cat, prediction for dog]   see train_generator.class_indices
        # Also you can see that sum of prediction for each class equals 1 due to softmax activation
        predicted = custom_model.predict(np.asarray([image]))
        index = np.argmax(predicted[0], axis=0)   # get the index of predicted class
        predicted_answer = list(train_generator.class_indices.keys())[index]   # get the class name you predicted
        print('\n\n\n')
        print('Predict this image as \"{}\"\n'.format(predicted_answer))
        # print out Probability of each label's node
        for label in range(len(train_generator.class_indices)):
            print('The probability that this is a {} is: {:.8f}\n'.format(list(train_generator.class_indices.keys())[label], predicted[0][label]))
        cv2.imshow('Image ' + str(i), image)
        cv2.waitKey()
        # cv2.destroyAllWindows()

cv2.destroyAllWindows()

###########################################################

# (method-2) testing an image
# testing an image with cat & dog receive bad result
# gives the full path to img_path.
# mind that filename extension need to be added. some imgs are jpg, others are jpeg
img_path = r'D:\s1040328\input_pipeline\data\test\test_cat_dog_2.jpg'


def preprocess_image(path=img_path):
    img = Image.open(path)
    img = img.resize((img_height, img_width), Image.BICUBIC)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    x = np.asarray(img)   # object to ndarray
    x = x / 255.
    batch_x = np.asarray([x])   # create for a batch [1, img_width, img_height, channel]
    return batch_x


img = preprocess_image(img_path)
mth2_predicted = custom_model.predict(img)
mth2_index = np.argmax(predicted[0], axis=0)
mth2_predicted_answer = list(train_generator.class_indices.keys())[mth2_index]   # get the class name you predicted
print('\n\n\n')
print('Predict this image as \"{}\"\n'.format(mth2_predicted_answer))
for label in range(len(train_generator.class_indices)):
    print('The probability that this is a {} is: {:.8f}\n'.format(list(train_generator.class_indices.keys())[label],
                                                                  mth2_predicted[0][label]))
image_for_cv2 = img[0][:, :, ::-1].copy()
cv2.imshow('Image ' + str(i), image_for_cv2)
cv2.waitKey()
cv2.destroyAllWindows()

###########################################################

# (method-3) predict your test automatically and saved to the result_path

result_path = r'D:\s1040328\input_pipeline\data\test_result'   # gives a full directory path to result_path
sub_file_prefix = 'predicted_as_'   # predict folder prefix
test_generator.reset()   # prepare to use test_generator entirely


# do not open files in the result_path, it will cause Permission error and OS error
# clean=True means that delete all the files in the sub_folder, so you will only get a batch of image prediction
# otherwise the predicted image will just add into the sub_folder
def predicted_each_batch_and_saved(clean=True, save_path=None, sub=None, batch_index=0):
    test_generator.batch_index = batch_index

    for category_name in train_generator.class_indices.keys():
        sub_folder = save_path + '\\' + sub + category_name
        if clean:
            if os.path.exists(sub_folder):
                shutil.rmtree(sub_folder, ignore_errors=False)
            os.makedirs(sub_folder)
            print('Clean all the folders and directories inside the save_path.')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        print('Create sub folders {} for prediction.'.format(sub_folder))
    for count, per_image in enumerate(test_generator.next(), 1):
        mth3_predicted = custom_model.predict(np.asarray([per_image]))
        mth3_index = np.argmax(mth3_predicted[0], axis=0)
        mth3_predicted_answer = list(train_generator.class_indices.keys())[mth3_index]
        for dir in os.listdir(save_path):
            if dir[len(sub_file_prefix):].find(mth3_predicted_answer):
                cv2.imwrite(save_path + '\\' + dir + '\\batch' + str(test_generator.batch_index-1) + '_' + str(count)
                            + '_' + str(mth3_predicted[0][mth3_index]) + '.jpg', per_image*255)


# example for iterate test_batch 0-9 & saving each image in every batch
for i in range(10):
    predicted_each_batch_and_saved(clean=False, save_path=result_path, sub=sub_file_prefix, batch_index=i)

# example for saving only one specific batch
predicted_each_batch_and_saved(clean=False, save_path=result_path, sub=sub_file_prefix, batch_index=15)


# keras github : https://github.com/keras-team/keras
# keras documentation : https://faroit.github.io/keras-docs/2.0.2/
