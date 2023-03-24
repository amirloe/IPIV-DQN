from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute,Concatenate,Input
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from keras.initializers import Constant
from losses import ipiv_dqn_loss, pi_loss, v_loss
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

def getconv_vanilla_model(input_shape,nb_actions):
    model = Sequential()
    print(K.image_data_format())
    if K.image_data_format() == 'channels_last':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))
    elif K.image_data_format() == 'channels_first':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=input_shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())
    model.compile(Adam(lr=1e-3), metrics=['mae'])
    return model

def getconv_ipiv_model(input_shape,nb_actions,bias_init):
        
    """
    Builds the Q-network model.
    """
    # Neural Net for Deep-Q learning Model
    inputs = Input(shape=input_shape, name='State_input')
    if K.image_data_format() == 'channels_last':
        # (width, height, channels)
        x = Permute((2, 3, 1), input_shape=input_shape)(inputs)
    elif K.image_data_format() == 'channels_first':
        # (channels, width, height)
        x = Permute((1, 2, 3), input_shape=input_shape)(inputs)
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    x = Convolution2D(32, (8, 8), strides=(4, 4))(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, (4, 4), strides=(2, 2))(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1))(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    pi = Dense(2*nb_actions, activation='linear',kernel_initializer=tf.keras.initializers.Zeros(),
                                    bias_initializer=Constant(value=bias_init), name='pi')(x) # pi initialization using bias
    q = Dense(nb_actions, activation="sigmoid", kernel_initializer='he_uniform', name='q')(x)
    piven_out = Concatenate(name='piven_out')([pi, q])
    model = Model(inputs=inputs, outputs=[piven_out], name='piven_model')
    # compile
    opt = tf.keras.optimizers.Adam(lr=0.00025,clipvalue=.6,clipnorm=1.)
    lossf = ipiv_dqn_loss(beta=0.9,lambda_in=15.0,alpha=0.1,action_size=nb_actions)
    # lossf = piven_dqn_loss_normal(beta=self.piven_beta,lambda_in=15.0,alpha=.25,action_size=self.action_size)
    pi_lossf = pi_loss(beta=1.,lambda_in=15.0,alpha=.25,action_size=nb_actions)
    v_lossf = v_loss(beta=0.,lambda_in=15.0,alpha=.25,action_size=nb_actions)
    model.compile(loss=lossf,metrics=[pi_lossf,v_lossf],optimizer=opt)
    return model