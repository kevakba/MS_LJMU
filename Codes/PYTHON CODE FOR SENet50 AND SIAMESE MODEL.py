# downloading the SENet50 pre-trained on vggface dataset 
!sudo pip install git+https://github.com/rcmalli/keras-vggface.git

# check installation
!pip show keras-vggface
!pip install keras_applications

# importing the required libraries
import keras.applications
from keras_vggface.vggface import VGGFace
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import tensorflow as tf


# loading the SENet50 model
model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# setting the model layers non-trainable
for layer in model.layers:
  layer.trainable = False

'''
freezing the BatchNormalization layer weights, as per practice in transfer learning
refer: https://keras.io/guides/transfer_learning/
'''
for layer in model.layers[-70:]:
  if 'BatchNormalization' in str(layer):
    layer.trainable = False
  else:
    layer.trainable = True

# defining input layer of model
inp = Input(shape=(224, 224,3))
y = model(inp, training=False)   # training=False inorder to keep bn layer in inference mode

#getting core of siamese network
core_model = Model(inputs=inp, outputs=y)

# defining the complete Siamese network
input_shape = (224,224,3)
input_image_1 = Input(input_shape)
input_image_2 = Input(input_shape)

y1 = core_model(input_image_1, training=False)   # training=False inorder to keep bn layer in inference mode
y2 = core_model(input_image_2, training=False)   # training=False inorder to keep bn layer in inference mode

# adding the distance layer at the end
l2_distance_layer = Lambda(
            lambda tensors: tf.math.reduce_sum(tf.math.squared_difference(tf.math.l2_normalize(tensors[0], axis=1) , 
                                                       tf.math.l2_normalize(tensors[1], axis=1)), axis=1)
            )
l2_distance = l2_distance_layer([y1, y2])

# defining the overall Siamese Architecture     
siamese_model = Model(
            inputs=[input_image_1, input_image_2], 
            outputs=l2_distance
            )

# creating the custom loss function
def my_loss_fn(y_true, y_pred):
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)
    return y_true*y_pred**2 + (1-y_true)*tf.math.maximum(0.,(5.0-y_pred))**2

# model compilation
siamese_model.compile(loss= my_loss_fn, 
                 metrics= None,
                 optimizer=tf.keras.optimizers.SGD(1e-5))



