from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Lambda, AveragePooling2D, MaxPooling2D
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar100
from keras import metrics
import numpy as np
import os
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import functools
import math



model_type = 'ResNetm%dv%d' % (50, 1)
train_epoch = 30



def resnet_layer(inputs,
                 num_filters=64,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=100):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 64
    num_res_blocks = int((depth - 2) / 6)
    
    #inputs = Input(shape=input_shape)
    x = resnet_layer(input_shape)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=2)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    #model = Model(inputs=inputs, outputs=outputs)
    return outputs




# In[4]:


#LOAD CIFAR
(x_train, y_train),(x_test,y_test) = cifar100.load_data()
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)


insh = x_train.shape[1:]
print(insh)


def preresnet_v1(input_shape=insh, depth=50):
    
    X_input = Input(input_shape)
    X = Lambda(lambda image: keras.backend.resize_images(image, 1,1,"channels_last" ))(X_input)

    X = Conv2D(64, (3,3), strides=(2,2),padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    Y = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(X)
    Y1 = resnet_v1(Y, depth = 50)
    
    model = Model(inputs=X_input, outputs = Y1, name='Resnet50')
    
    
    return model

model = preresnet_v1(input_shape=insh, depth =50)

# print(x_validate.shape)

# training parameters
batch_size = 150
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr



top1_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=1)
top1_acc.__name__= 'top1_acc'

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy', metrics.top_k_categorical_accuracy, top1_acc])
model.summary()


print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'Cifar100_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=0.1,
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]
print(callbacks)
data_augmentation = False


# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=train_epoch,
              validation_split=(10/50),
              shuffle=True,
              callbacks=callbacks)

     

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

fig, (vax, hax) = plt.subplots(2,1)
y1 = history.history['loss']
y2 = history.history['val_loss']

y3 = history.history['accuracy']
y4 = history.history['val_accuracy']

vax.plot(y1,label='trainingloss')
vax.plot(y2,label='validationlosses')

vax.set_xlabel('Epoch')
vax.set_ylabel('Loss')
vax.legend(loc=3)
vax.grid(True)

hax.plot(y3,label='training_acc')
hax.plot(y4,label='validation_acc')

hax.set_xlabel('Epoch')
hax.set_ylabel('Accuracy')
hax.legend(loc=2)
hax.grid(True)

fig.suptitle("CIFAR100 DATA WITH RESNET 50 ARCHITECTURE")

if not os.path.isdir('results'):
    os.mkdir('results')
if not os.path.isdir('results'):
    os.mkdir('results')

p = 'results/CIFAR100_RESNET' + '.png'

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(p)

plt.show()

model.save('cifar100_resnet50')



def cifar_grid(X,Y,inds,n_col,predictions=None):
    print(n_col)
    print(inds)

    N = len(inds)
    n_row = int(math.ceil(1.0*N/n_col))
    fig, axes = plt.subplots(n_row,n_col,figsize=(10,10))
  
    print()
    for j in range(n_row):
        for k in range(n_col):
            i_inds = j*n_col+k
            #print("i_inds", i_inds)
            i_data = inds[i_inds]
   
      
            axes[j][k].set_axis_off()
            if i_inds < N:
                axes[j][k].imshow(X[i_data,...], interpolation="nearest")
                label = np.argmax(Y[i_data,...])
                print("label is", label)
                if predictions is not None:
                    pred = np.argmax(predictions[i_data,...])
                   # print(pred)
                    if label != pred:
                        pred = "Wrong prediction"
                        axes[j][k].set_title(pred, color="red")
    fig.set_tight_layout(True)
    return fig



datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)


predict_gen = model.predict_generator(datagen.flow(x_test, y_test,
    batch_size=batch_size, shuffle=False),
    steps=(x_test.shape[0] // batch_size)+1, workers=4)


indices = [np.random.choice(range(len(x_test))) 
           for i in range(20)]

#labels = [np.random.choice(range(len(y_test))) 
         #  for i in range(20)]


cifar_grid(x_test,y_test,indices,5,predictions=predict_gen)

p = 'results/Predictions' + '.png'

plt.savefig(p)

plt.show()


