import tensorflow as tf
from keras.layers import Dense
from keras import Input
from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Add, \
    GlobalAveragePooling2D


def generate_resnet50(image_size, include_top=True, classes=1000):
    """
    Generates a ResNet50 network from basic Keras layers. This is a challenge set in the
    Advanced Computer Vision course on Udemy by Lazy Programmer. The ResNet nets can also
    be created using the built-in Keras Applications, this is simply a learning exercise.
    ```python
    # Test code
    model = generate_resnet50([320, 240, 3], classes=4)
    model.compile()
    model.summary(line_length=150)
    ```
    :param image_size: [width, height] of the training and test images.
    :param include_top: Set to True if you want to include the Dense layer after the convolutions.
    :param classes: The number of classes to output from the Dense layer.
    :return: A Model object of the created network ready fro compilation.
    """
    def build_input_section(input_layer):
        """
        Builds the input section of the ResNet50 network.
        :param input_layer: A reference to the Keras Input.
        :return: Returns a Tensorlike object containing the layers that make up the input section conv1 and pool1.
        """
        c1_x = ZeroPadding2D(padding=3, name='conv1_pad')(input_layer)
        c1_x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, name='conv1_conv')(c1_x)
        c1_x = BatchNormalization(name='conv1_bn')(c1_x)
        c1_x = ReLU(name='conv1_relu')(c1_x)
        c1_x = ZeroPadding2D(padding=1, name='pool1_pad')(c1_x)
        c1_x = MaxPooling2D(pool_size=3, strides=2, name='pool1_pool')(c1_x)
        return c1_x

    def build_resnet_block(x_in, conv_id, block_id, filters_in, filters_out, initial_stride=1):
        """
        Builds a resnet50 block consisting of a:
        1 x 1 Conv2D, Batch Normalisation, Relu
        3 x 3 Conv2D, Batch Normalisation, Relu
        1 x 1 Conv2D, Batch Normalisation
        This is then summed with the input tensor and if the input tensor has a different depth, a 1 x 1 Conv2D
        and Batch Normalisation is inserted before adding.
        Finally, a ReLU activation layer is added before the whole block is returned.
        :param x_in: The input tensor from the previous block.
        :param conv_id: The layer number used in naming (see the paper's Table 1. layer_name column)
        :param block_id: The block number used in naming
        :param filters_in: The number of filters to use for block 1 and block 2.
        :param filters_out: The number of filters to use for block 3 and the output.
        :param initial_stride: The stride to use to reduce the image size.
        :return: A tensorlike that can be added to the Model.
        """
        # residual pathway
        conv_x = Conv2D(filters=filters_in, kernel_size=1, strides=initial_stride, padding='same',
                        name=f'conv{conv_id}_block{block_id}_1_conv')(x_in)
        conv_x = BatchNormalization(name=f'conv{conv_id}_block{block_id}_1_bn')(conv_x)
        conv_x = ReLU(name=f'conv{conv_id}_block{block_id}_1_relu')(conv_x)
        conv_x = Conv2D(filters=filters_in, kernel_size=3, padding='same',
                        name=f'conv{conv_id}_block{block_id}_2_conv')(conv_x)
        conv_x = BatchNormalization(name=f'conv{conv_id}_block{block_id}_2_bn')(conv_x)
        conv_x = ReLU(name=f'conv{conv_id}_block{block_id}_2_relu')(conv_x)
        conv_x = Conv2D(filters=filters_out, kernel_size=1, padding='same',
                        name=f'conv{conv_id}_block{block_id}_3_conv')(conv_x)
        conv_x = BatchNormalization(name=f'conv{conv_id}_block{block_id}_3_bn')(conv_x)

        # this is the identity pathway which only needs to convolve if the number of channels has changed
        id_src = x_in
        if id_src.shape[-1] != conv_x.shape[-1]:
            id_src = Conv2D(filters_out, kernel_size=1, strides=initial_stride, padding='same',
                            name=f'conv{conv_id}_block{block_id}_0_conv')(x_in)
            id_src = BatchNormalization(name=f'conv{conv_id}_block{block_id}_0_bn')(id_src)

        # summing and activation
        conv_x = Add(name=f'conv{conv_id}_block{block_id}_add')([conv_x, id_src])
        conv_x = ReLU(name=f'conv{conv_id}_block{block_id}_out')(conv_x)
        return conv_x

    # conv1
    i = Input(shape=image_size)
    x = build_input_section(i)

    # conv2.x
    x = build_resnet_block(x_in=x, filters_in=64, filters_out=256, conv_id=2, block_id=1)
    x = build_resnet_block(x_in=x, filters_in=64, filters_out=256, conv_id=2, block_id=2)
    x = build_resnet_block(x_in=x, filters_in=64, filters_out=256, conv_id=2, block_id=3)

    # conv3.x
    x = build_resnet_block(x_in=x, filters_in=128, filters_out=512, conv_id=3, block_id=1, initial_stride=2)
    x = build_resnet_block(x_in=x, filters_in=128, filters_out=512, conv_id=3, block_id=2)
    x = build_resnet_block(x_in=x, filters_in=128, filters_out=512, conv_id=3, block_id=3)
    x = build_resnet_block(x_in=x, filters_in=128, filters_out=512, conv_id=3, block_id=4)

    # conv4.x
    x = build_resnet_block(x_in=x, filters_in=256, filters_out=1024, conv_id=4, block_id=1, initial_stride=2)
    x = build_resnet_block(x_in=x, filters_in=256, filters_out=1024, conv_id=4, block_id=2)
    x = build_resnet_block(x_in=x, filters_in=256, filters_out=1024, conv_id=4, block_id=3)
    x = build_resnet_block(x_in=x, filters_in=256, filters_out=1024, conv_id=4, block_id=4)
    x = build_resnet_block(x_in=x, filters_in=256, filters_out=1024, conv_id=4, block_id=5)
    x = build_resnet_block(x_in=x, filters_in=256, filters_out=1024, conv_id=4, block_id=6)

    # conv5.x
    x = build_resnet_block(x_in=x, filters_in=512, filters_out=2048, conv_id=5, block_id=1, initial_stride=2)
    x = build_resnet_block(x_in=x, filters_in=512, filters_out=2048, conv_id=5, block_id=2)
    x = build_resnet_block(x_in=x, filters_in=512, filters_out=2048, conv_id=5, block_id=3)

    x = GlobalAveragePooling2D(name='avg_pool')(x)

    # add the dense layer
    if include_top:
        x = Dense(classes, activation='relu')(x)

    return tf.keras.models.Model(i, x)


