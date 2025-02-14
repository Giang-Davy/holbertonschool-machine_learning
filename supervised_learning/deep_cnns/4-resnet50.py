#!/usr/bin/env python3
"""fonction"""


from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Création du réseau ResNet"""
    input = K.layers.Input(shape=(224, 224, 3))
    initializer = K.initializers.HeNormal(seed=0)

    # Conv1
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        padding="same",
        strides=(2,2),
        kernel_initializer=initializer)(input)
    batch_norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    activation1 = K.layers.Activation('relu')(batch_norm1)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="same")(activation1)

    # Block 1 (conv2)
    projection_block1 = projection_block(pool1, [64, 64, 256], s=1)
    id_block1_2 = identity_block(projection_block1, [64, 64, 256])
    id_block1_3 = identity_block(id_block1_2, [64, 64, 256])

    # Projection Block ici pour ajuster la taille (de 256 à 512)
    proj_block2 = projection_block(id_block1_3, [128, 128, 512], s=2)
    id_block2_1 = identity_block(proj_block2, [128, 128, 512])
    id_block2_2 = identity_block(id_block2_1, [128, 128, 512])
    id_block2_3 = identity_block(id_block2_2, [128, 128, 512])

    # Block 3 (conv4)
    proj_block3 = projection_block(id_block2_3, [256, 256, 1024], s=2)
    id_block3_1 = identity_block(proj_block3, [256, 256, 1024])
    id_block3_2 = identity_block(id_block3_1, [256, 256, 1024])
    id_block3_3 = identity_block(id_block3_2, [256, 256, 1024])
    id_block3_4 = identity_block(id_block3_3, [256, 256, 1024])
    id_block3_5 = identity_block(id_block3_4, [256, 256, 1024])

    # Block 4 (conv5)
    proj_block4 = projection_block(id_block3_5, [512, 512, 2048], s=2)
    id_block4_1 = identity_block(proj_block4, [512, 512, 2048])
    id_block4_2 = identity_block(id_block4_1, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D((7, 7), strides=(1, 1))(id_block4_2)

    output_layer = K.layers.Dense(1000, activation='softmax',
                                  kernel_initializer=initializer)(avg_pool)

    model = K.models.Model(inputs=input, outputs=output_layer)

    return model
