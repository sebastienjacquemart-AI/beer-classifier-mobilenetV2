#!/usr/bin/env python3

"""Model to classify draft beers

This file contains all the model information: the training steps, the batch
size and the model itself.
"""

import tensorflow as tf

def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value.
    """
    return 16

def get_epochs():
    """Returns number of epochs that will be used by your solution.
    It is recommended to change this value.
    """
    return 100

def solution(input_layer):
    """Returns a compiled model.

    This function is expected to return a model to identity the different beers.
    The model's outputs are expected to be probabilities for the classes and
    and it should be ready for training.
    The input layer specifies the shape of the images. The preprocessing
    applied to the images is specified in data.py.

    Add your solution below.

    Parameters:
        input_layer: A tf.keras.layers.InputLayer() specifying the shape of the input.
            RGB colored images, shape: (width, height, 3)
    Returns:
        model: A compiled model
    """
    # INITIALIZE VARIABLES
    num_classes = 5
    fine_tuning = False
    fine_tune_at = 100

    # DATA AUGMENTATION
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal",input_shape=input_layer.shape[-3:]),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            #tf.keras.layers.RandomBrightness(0.1)
        ]
    )

    # MODELS FROM SCRATCH
    model_1 = tf.keras.Sequential([
        input_layer,
        data_augmentation,
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model_2 = tf.keras.Sequential([
        input_layer,
        data_augmentation,
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    # MODELS USING TRANSFER LEARNING
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_layer.shape[-3:],
                                               include_top=False,
                                               weights='imagenet')
    if not fine_tuning:
        base_model.trainable = False
    else:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    
    model_3 = tf.keras.Sequential([
        input_layer,
        data_augmentation,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    """
    x = data_augmentation(input_layer)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model_3 = tf.keras.Model(input_layer, output_layer)
    """
    # CHOOSE MODEL
    model = model_3

    # COMPILE MODEL
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

    return model

'''
base_model = tf.keras.applications.MobileNetV2(input_shape=input_layer.shape[-3:],
                                               include_top=False,
                                               weights='imagenet')
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(5, activation='softmax')

    x = data_augmentation(input_layer)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    #x = tf.keras.layers.Dropout(0.2)(x)
    output_layer = prediction_layer(x)

        model = tf.keras.Model(input_layer, output_layer)

'''
"""
    model_3 = tf.keras.Sequential([
        input_layer,
        data_augmentation,
        tf.keras.layers.Lambda(lambda x: base_model(x, training=False)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
"""