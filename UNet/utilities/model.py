import tensorflow as tf


def ConvBlock(input, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    return x

def EncoderBlock(input, num_filters):
    x = ConvBlock(input, num_filters)
    p = tf.keras.layers.MaxPooling2D((2, 2))(x)
    return x, p

def DecoderBlock(input, skip_features, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = ConvBlock(x, num_filters)
    return x

def build_model(input_shape):
    inputs = tf.keras.layers.Input(input_shape)

    s1, p1 = EncoderBlock(inputs, 64)
    s2, p2 = EncoderBlock(p1, 128)
    s3, p3 = EncoderBlock(p2, 256)
    s4, p4 = EncoderBlock(p3, 512)

    b1 = ConvBlock(p4, 1024)

    d1 = DecoderBlock(b1, s4, 512)
    d2 = DecoderBlock(d1, s3, 256)
    d3 = DecoderBlock(d2, s2, 128)
    d4 = DecoderBlock(d3, s1, 64)

    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = tf.keras.Model(inputs, outputs, name="U-Net")
    return model

# if __name__ == "__main__":
#     input_shape = (256, 256, 1)
#     model = build_model(input_shape)
#     model.summary()