import os
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.applications import *


class MyModel(tf.keras.Model):

    IMAGE_SIZE = (224, 224)

    METRICS = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'), 
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
    
    models = {"mobilenet":  (mobilenet_v3.preprocess_input, MobileNetV3Large),
              "xception":   (xception.preprocess_input, Xception),
              "resnet": (resnet_v2.preprocess_input, resnet_v2.ResNet50V2) }


    @classmethod
    def create_standard_version(cls, load_weights_path = None, compile = False):
        model = MyModel(network_model = "mobilenet", pooling = "max", extra_cols_out = 128, dense_intermediate = 256)

        if load_weights_path is not None:
            model.load_weights(load_weights_path)

        if compile:
            model.compile( loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=cls.METRICS)

        return model

    def __init__(self,  network_model = "mobilenet",
                        pooling = "max",
                        dense_intermediate = -1,
                        dropout = 0.5,
                        extra_cols_out = 32,
                        bias = tf.keras.initializers.Constant(np.log([500 / 30000]))
                ):
                        
        super().__init__()
        preprocessor_and_network = self.models[network_model]
        self.image_preprocessor = preprocessor_and_network[0]
        self.image_feature_extractor = preprocessor_and_network[1](weights='imagenet', include_top = False, pooling = pooling)

        self.image_size = self.IMAGE_SIZE

        self.extra_cols_dense = tf.keras.layers.Dense(extra_cols_out, activation = "relu", name="extra_cols_dense")
        self.extra_cols_flatten = tf.keras.layers.Flatten()

        self.use_intermediate = dense_intermediate > 0
        if self.use_intermediate:
            self.dropout1 = tf.keras.layers.Dropout(dropout)
            self.dense1 = tf.keras.layers.Dense(dense_intermediate, activation="relu", name = "final_dense_intermediate")

        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.out = tf.keras.layers.Dense(1, activation = "sigmoid", bias_initializer=bias, name = "final_dense")

    def preprocess_images(self, images):
        x = self.image_preprocessor(images)
        x = tf.image.resize(x, self.image_size)
        return x

    def call(self, inputs):
        preprocessed_images = self.preprocess_images(inputs['image'])
        image_output = self.image_feature_extractor(preprocessed_images, training = False)

        data = inputs['data']
        data = self.extra_cols_dense(data)
        data = self.extra_cols_flatten(data)

        x = tf.concat([data, image_output], axis = -1)
        if self.use_intermediate:
            x = self.dropout1(x)
            x = self.dense1(x)
            
        x = self.dropout2(x)
        out = self.out(x)
        
        return out

