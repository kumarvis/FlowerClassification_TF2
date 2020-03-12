import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import project_params as pp

def get_custom_model():
    input_shape = pp.input_shape
    resnet50_feature_extractor = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights='imagenet', input_shape=input_shape)

    frozen_layers, trainable_layers = [], []
    for layer in resnet50_feature_extractor.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.trainable = False
            frozen_layers.append(layer.name)
        else:
            if len(layer.trainable_weights) > 0:
                # We list as "trainable" only the layers with trainable parameters.
                trainable_layers.append(layer.name)

    num_classes = pp.num_classes
    features = resnet50_feature_extractor.output
    avg_pool = GlobalAveragePooling2D(data_format='channels_last')(features)
    predictions = Dense(num_classes, activation='softmax')(avg_pool)
    resnet50_freeze = Model(resnet50_feature_extractor.input, predictions)
    return resnet50_freeze