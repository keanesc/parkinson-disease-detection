import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(num_classes):
    base_model1 = keras.applications.EfficientNetB4(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model2 = keras.applications.ViTBase16(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )

    base_model1.trainable = False
    base_model2.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))

    x1 = base_model1(inputs)
    x1 = layers.GlobalAveragePooling2D()(x1)

    x2 = base_model2(inputs)
    x2 = layers.GlobalAveragePooling2D()(x2)

    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(
        x
    )  # Multi-class classification

    model = keras.Model(inputs, outputs)
    return model
