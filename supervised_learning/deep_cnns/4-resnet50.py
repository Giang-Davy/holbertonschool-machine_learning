#!/usr/bin/env python3


from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet_network():
    """Création du réseau ResNet"""
    input = K.layers.Input(shape=(224, 224, 3))
    initializer = K.initializers.VarianceScaling(scale=2.0)

    # Conv1
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        activation='relu',
        padding="same",
        strides=2,
        kernel_initializer=initializer)(input)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="same")(conv1)

    # Block 1 (conv2)
    id_block1_1 = identity_block(pool1, [64, 64, 256])
    id_block1_2 = identity_block(id_block1_1, [64, 64, 256])
    id_block1_3 = identity_block(id_block1_2, [64, 64, 256])

    # Block 2 (conv3)
    proj_block2 = projection_block(id_block1_3, [128, 128, 512], s=2)
    id_block2_1 = identity_block(proj_block2, [128, 128, 512])
    id_block2_2 = identity_block(id_block2_1, [128, 128, 512])
    id_block2_3 = identity_block(id_block2_2, [128, 128, 512])
    id_block2_4 = identity_block(id_block2_3, [128, 128, 512])

    # Block 3 (conv4)
    proj_block3 = projection_block(id_block2_4, [256, 256, 1024], s=2)
    id_block3_1 = identity_block(proj_block3, [256, 256, 1024])
    id_block3_2 = identity_block(id_block3_1, [256, 256, 1024])
    id_block3_3 = identity_block(id_block3_2, [256, 256, 1024])
    id_block3_4 = identity_block(id_block3_3, [256, 256, 1024])
    id_block3_5 = identity_block(id_block3_4, [256, 256, 1024])
    id_block3_6 = identity_block(id_block3_5, [256, 256, 1024])

    # Block 4 (conv5)
    proj_block4 = projection_block(id_block3_6, [512, 512, 2048], s=2)
    id_block4_1 = identity_block(proj_block4, [512, 512, 2048])
    id_block4_2 = identity_block(id_block4_1, [512, 512, 2048])
    id_block4_3 = identity_block(id_block4_2, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D((7, 7), strides=(1, 1))(id_block4_3)
    dropout = K.layers.Dropout(0.4)(avg_pool)
    output_layer = K.layers.Dense(1000, activation='softmax')(dropout)

    model = K.models.Model(inputs=input, outputs=output_layer)

    return model
