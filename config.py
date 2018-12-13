import tensorflow as tf

flags = tf.app.flags

############################
#    Hyper Parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')

# for training
flags.DEFINE_integer('batch_size', 200, 'batch size')
flags.DEFINE_integer('epoch', 20, 'epoch')
flags.DEFINE_integer('num_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.392, 'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')

############################
#   Datasets Parameters    #
############################

flags.DEFINE_integer('max_features', 5000, 'max_features size')
flags.DEFINE_integer('max_len', 400, 'the maximum length of the sentence')
flags.DEFINE_integer('embed_dim', 50, 'the size of the embedding')

############################
#   Environment Setting    #
############################
flags.DEFINE_string('dataset', 'imdb', 'The name of dataset [imdb, polaritydata]')
flags.DEFINE_string('result', 'results', 'path for saving results')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')

cfg = tf.app.flags.FLAGS
