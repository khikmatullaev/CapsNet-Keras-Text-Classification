from keras import layers, models
from capsule_layers import CapsuleLayer, PrimaryCap, Length, Mask


def CapsNet(input_shape, n_class, num_routing, vocab_size, embed_dim, max_len):
    """
    :param input_shape: data shape, 4d, [None, width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :param vocab_size:
    :param embed_dim:
    :param max_len:
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=input_shape)
    embed = layers.Embedding(vocab_size, embed_dim, input_length=max_len)(x)

    conv1 = layers.Conv1D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(embed)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primary_caps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps")

    # Layer 3: Capsule layer. Routing algorithm works here.
    category_caps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='category_caps')(primary_caps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')(category_caps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked = Mask()([category_caps, y])  # The true label is used to mask the output of capsule layer.
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(max_len, activation='sigmoid')(x_recon)
    # x_recon = layers.Reshape(target_shape=[1], name='out_recon')(x_recon)

    # two-input-two-output keras Model
    return models.Model([x, y], [out_caps, x_recon])