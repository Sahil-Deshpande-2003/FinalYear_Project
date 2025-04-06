import os
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Layer, Add  # For layer-based operations
from keras import backend as K
from keras import regularizers, initializers
# from keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping
from keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# Custom X_plus Layer
class X_plus_Layer(Layer):
    def __init__(self, **kwargs):
        super(X_plus_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', initializer='zeros', trainable=True)
        super(X_plus_Layer, self).build(input_shape)

    def call(self, inpt_x):
        x, A = inpt_x

        # Repeat x to match dimensions
        x_diag = x
        for _ in range(24):
            x_diag = K.concatenate([x_diag, x], axis=2)
        x_diag = K.expand_dims(x_diag, axis=3)
        x_diag_channals = x_diag
        for _ in range(2):
            x_diag_channals = K.concatenate([x_diag_channals, x_diag], axis=3)

        x_mask = x
        for _ in range(24):
            x_mask = K.concatenate([x_mask, x], axis=2)
        x_mask = K.expand_dims(x_mask, axis=3)
        x_mask_channals = x_mask
        for _ in range(2):
            x_mask_channals = K.concatenate([x_mask_channals, x_mask], axis=3)

        # a_part = self.alpha * K.multiply(x_diag_channals, A) # CHECKING!!!
        a_part = self.alpha * tf.multiply(x_diag_channals, A)
        b_part = self.beta * x_mask_channals

        ans = Add()([a_part, b_part])
        return ans

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 300, 25, 3)


# # Attention Map Function
# def attention_map(inpt, channels):
#     s_o = inpt[0]  # (batch, width=25)
#     t_o = inpt[1]  # (batch, height=300)
#     ipt = inpt[2]  # (batch, height=300, width=25, channels=3)

#     height = 300
#     width = 25

#     # Spatial attention
#     s_o = K.l2_normalize(s_o, axis=1)
#     s_map = K.expand_dims(s_o, axis=1)
#     for _ in range(height - 1):
#         s_map = K.concatenate([s_map, K.expand_dims(s_o, axis=1)], axis=1)

#     # Temporal attention
#     t_o = K.l2_normalize(t_o, axis=1)
#     t_map = K.expand_dims(t_o, axis=2)
#     for _ in range(width - 1):
#         t_map = K.concatenate([t_map, K.expand_dims(t_o, axis=2)], axis=2)

#     # Element-wise multiply for attention
#     a_o = K.multiply(s_map, t_map)

#     # ROI mask (same for all batches)
#     roi_mask = np.zeros((300, 25))
#     for i in range(300):
#         for j in [3, 8, 12, 14, 17, 19]:
#             roi_mask[i, j] = 1
#     roi_mask = tf.constant(roi_mask, dtype=tf.float32)
#     a_o = a_o + roi_mask  # Broadcast over batch

#     # Expand for channels
#     a_o = K.expand_dims(a_o, axis=3)
#     a_map = a_o
#     for _ in range(channels - 1):
#         a_map = K.concatenate([a_map, a_o], axis=3)

#     return K.multiply(a_map, ipt)

# class AttentionMapLayer(Layer):
#     def __init__(self, channels, **kwargs):
#         self.channels = channels
#         super(AttentionMapLayer, self).__init__(**kwargs)

#     def call(self, inputs):
#         s_o, t_o, ipt = inputs
#         height = 300
#         width = 25

#         s_o = K.l2_normalize(s_o, axis=1)
#         s_map = K.expand_dims(s_o, axis=1)
#         for _ in range(height - 1):
#             s_map = K.concatenate([s_map, K.expand_dims(s_o, axis=1)], axis=1)

#         t_o = K.l2_normalize(t_o, axis=1)
#         t_map = K.expand_dims(t_o, axis=2)
#         for _ in range(width - 1):
#             t_map = K.concatenate([t_map, K.expand_dims(t_o, axis=2)], axis=2)

#         a_o = K.multiply(s_map, t_map)

#         roi_mask = np.zeros((300, 25))
#         for i in range(300):
#             for j in [3, 8, 12, 14, 17, 19]:
#                 roi_mask[i, j] = 1
#         roi_mask = tf.constant(roi_mask, dtype=tf.float32)
#         a_o = a_o + roi_mask

#         a_o = K.expand_dims(a_o, axis=3)
#         a_map = a_o
#         for _ in range(self.channels - 1):
#             a_map = K.concatenate([a_map, a_o], axis=3)

#         return K.multiply(a_map, ipt)

#     def compute_output_shape(self, input_shape):
#         return input_shape[2]  # Same as the input image tensor

#     def get_config(self):
#         config = super(AttentionMapLayer, self).get_config()
#         config.update({'channels': self.channels})
#         return config

# def attention_map(inpt, channels):
#     s_o = inpt[0]  # shape (batch_size, width=25)
#     t_o = inpt[1]  # shape (batch_size, height=300)
#     ipt = inpt[2]   # shape (batch_size, height=300, width=25, channels=3)

#     height = 300
#     width = 25
    
#     ''' adaptive spatial attention '''
#     s_o = K.l2_normalize(s_o, axis=1)
#     s_map = K.expand_dims(s_o, axis=1)  # shape (batch_size, 1, width)
#     s_o = K.expand_dims(s_o, axis=1)
#     for h in range(height-1):
#         s_map = K.concatenate([s_map, s_o], axis=1)  # shape (batch_size, height, width)

#     ''' frame-level temporal attention '''
#     t_o = K.l2_normalize(t_o, axis=1)
#     t_map = K.expand_dims(t_o, axis=2)  # shape (batch_size, height, 1)
#     t_o = K.expand_dims(t_o, axis=2)
#     for w in range(width-1):
#         t_map = K.concatenate([t_map, t_o], axis=2)  # shape (batch_size, height, width)

#     ''' Prior Spatial Attention '''
#     a_o = multiply([s_map, t_map])  # shape (batch_size, height, width)
    
#     # Create ROI mask (now without batch dimension)
#     roi_mask = np.zeros((300, 25))  # shape (height, width)
#     for i in range(300):
#         for j in [3,8,12,14,17,19]:
#             roi_mask[i, j] = 1
    
#     # Convert to tensor and let broadcasting handle batch dimension
#     roi_mask = tf.constant(roi_mask, dtype=tf.float32)  # shape (300,25)
#     a_o = a_o + roi_mask  # Automatically broadcasts to (batch_size,300,25)
    
#     ''' Expand for channels '''
#     a_map = K.expand_dims(a_o, axis=3)  # shape (batch_size, height, width, 1)
#     a_o = K.expand_dims(a_o, axis=3)
#     for c in range(channels-1):
#         a_map = K.concatenate([a_map, a_o], axis=3)  # shape (batch_size, height, width, channels)
    
#     out = multiply([a_map, ipt])  # element-wise multiply with input
#     return out

class AttentionMapLayer(Layer):
    def __init__(self, channels = 3, **kwargs):
        self.channels = channels
        super(AttentionMapLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        roi_map_value = np.zeros((1, 300, 25))
        for i in range(300):
            for j in [3, 8, 12, 14, 17, 19]:
                roi_map_value[0, i, j] = 1
        self.roi_map = tf.constant(roi_map_value, dtype=tf.float32)
        super(AttentionMapLayer, self).build(input_shape)

    def call(self, inputs):
        s_o, t_o, ipt = inputs

        height = 300
        width = 25
        channels = self.channels

        s_o = K.l2_normalize(s_o, axis=1)
        s_map = K.expand_dims(s_o, axis=1)
        s_o = K.expand_dims(s_o, axis=1)
        for h in range(height - 1):
            s_map = K.concatenate([s_map, s_o], axis=1)

        t_o = K.l2_normalize(t_o, axis=1)
        t_map = K.expand_dims(t_o, axis=2)
        t_o = K.expand_dims(t_o, axis=2)
        for w in range(width - 1):
            t_map = K.concatenate([t_map, t_o], axis=2)

        # a_o = multiply([s_map, t_map]) CHECKING!!!!!!!!!!!!!!!!!!
        a_o = tf.multiply(s_map, t_map)
        a_o = Add()([a_o, self.roi_map])
        a_map = K.expand_dims(a_o, axis=3)
        a_o = K.expand_dims(a_o, axis=3)
        for c in range(channels - 1):
            a_map = K.concatenate([a_map, a_o], axis=3)

        # return multiply([a_map, ipt])  CHECKING!!!!!!!!!!!!!!!!!!
        return tf.multiply(a_map, ipt)
  

    def compute_output_shape(self, input_shape):
        return (input_shape[2][0], 300, 25, self.channels)
    
    def get_config(self):
        config = super(AttentionMapLayer, self).get_config()
        config.update({'channels': self.channels})
        return config

# Load the model with custom objects
print("LIFE BEFORE MODEL")
# model = load_model(
#     "temp (1).h5",
#     custom_objects={'X_plus_Layer': X_plus_Layer, 'attention_map': attention_map}
# )

model = load_model("temp (1).h5", custom_objects={'X_plus_Layer': X_plus_Layer, 'AttentionMapLayer': AttentionMapLayer})

x_test_mit = np.load(r'EX_STORE\Beauty_app\dfdc\stmap\tmpaxy1a0e_.avi\tmpaxy1a0e_.avi.npy')
x_test_meso = np.load('Meso.npy')
x_test_mit = np.expand_dims(x_test_mit, axis=0)   # (1, 300, 25, 3)
x_test_meso = np.expand_dims(x_test_meso, axis=0) # (1, ...your meso shape...)


print(f"x_test_mit = {x_test_mit}")
print(f"x_test_meso = {x_test_meso}")

predictions = model.predict([x_test_mit, x_test_meso], batch_size=1, verbose=1)
# predictions = model.predict([x_test_mit, x_test_meso], verbose=1)

print(f"predictions = {predictions}")
