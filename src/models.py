from collections import deque
import keras
from keras.layers import Dense,Input,BatchNormalization,Concatenate
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential,Model
from keras.initializers import RandomNormal, Constant, GlorotNormal,he_uniform
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input,Concatenate
from keras.optimizers import Adam
from keras.models import Sequential,Model
from keras.initializers import RandomNormal, Constant, GlorotNormal,he_uniform
import keras.backend as K
import random
import tensorflow as tf
import numpy as np
import wandb
from wandb.keras import WandbCallback
from processing import process_observation,process_state_batch
import pdb

"""
Author: Amir Loewenthal

This file contains the model definitions for the DQN and PIVEN DQN models.
This file also contains the loss functions for the PIVEN DQN model.
Its main purpose is to be imported by the ddqn.py file.

"""


"""
This section contains the IPIV DQN loss functions.
As presented in the paper, the IPIV DQN loss function is a combination of  the PI loss function and the V loss function.
The PI loss function is a combination of the MPIW_capt and the PICP_soft.
The V loss function is the MSE between the predicted value and the true value. As used in DQN.
"""
def loss_normalize(loss):
    
    loss_value = tf.Variable(1.0)

    print(loss)
    loss_normalized = loss / loss_value.assign(loss)

    return loss_normalized

def piven_dqn_loss_normal(lambda_in=15., soften=160., alpha=0.05, beta=0.5 ,action_size=3):
    """
    :param lambda_in: lambda parameter
    :param soften: soften parameter
    :param alpha: confidence level (1-alpha)
    :param beta: balance parameter
    """

    def piven_loss(y_true, y_pred):

        # from Figure 1 in the paper
        y_U = {}
        y_L = {} # L(x)
        y_v = {} # v(x)
        y_T = {}  # y(x)
        for x in range(action_size):
            y_U[x] = y_pred[:, x]
            y_L[x] = y_pred[:, action_size + x]
            y_v[x] = y_pred[:, 2 * action_size + x]
            y_T[x] = y_true[:, x]
        
        N_ = {i:tf.cast(tf.size(y_T[i]), tf.float32) for i in range(action_size)}  # batch size

        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)

        # k_soft uses sigmoid
        k_soft = {i:tf.multiply(tf.sigmoid((y_U[i] - y_T[i]) * soften),
                             tf.sigmoid((y_T[i] - y_L[i]) * soften)) for i in range(action_size)}
        
    
        # k_hard uses sign step function
        k_hard = {i:tf.multiply(tf.maximum(0., tf.sign(y_U[i] - y_T[i])),
                             tf.maximum(0., tf.sign(y_T[i] - y_L[i]))) for i in range(action_size)}
       
        # MPIW_capt from equation 4
        MPIW_capt = {i:tf.divide(tf.reduce_sum(tf.abs(y_U[i] - y_L[i]) * k_hard[i]),
                              tf.reduce_sum(k_hard[i]) + 0.001) for i in range(action_size)}
        
 
        # equation 1 where k is k_soft
        PICP_soft = {i:tf.reduce_mean(k_soft[i]) for i in range(action_size)}
        

        # pi loss from section 4.2
        pi_loss = {i:MPIW_capt[i] + lambda_ * tf.sqrt(N_[i]) * tf.square(tf.maximum(0., 1. - alpha_ - PICP_soft[i])) for i in range(action_size)}
       
        y_piven = {i:y_v[i] * y_U[i] + (1 - y_v[i]) * y_L[i] for i in range(action_size)}  # equation 3
        y_piven = {i:tf.reshape(y_piven[i], (-1, 1)) for i in range(action_size)}
        y_T = {i:tf.reshape(y_T[i], (-1, 1)) for i in range(action_size)}
        v_loss = {i:tf.losses.mean_squared_error(y_T[i], y_piven[i]) for i in range(action_size)}# equation 5
        
        #Normalize
        # pi_loss={i:loss_normalize(pi_loss[i]) for i in range(action_size)}
        v_loss={i:loss_normalize(v_loss[i]) for i in range(action_size)}
        
        # piven_loss_ = (tf.sign(v_loss)) *(beta * pi_loss + (1 - beta) * v_loss)  # equation 6
        piven_loss_ = {i:(tf.sign(v_loss[i])) *(beta * pi_loss[i] + (1 - beta) * v_loss[i]) for i in range(action_size)}  # equation 6
        # piven_loss_ =  v_loss # equation 6
        loss_final = tf.concat([piven_loss_[i] for i in range(action_size)],axis=0)
        # print(v_loss)
        return loss_final

    return piven_loss


def v_loss(lambda_in=15., soften=160., alpha=0.05, beta=0. ,action_size=3):
    """

    :param lambda_in: lambda parameter
    :param soften: soften parameter
    :param alpha: confidence level (1-alpha)
    :param beta: balance parameter
    """

    def v_loss(y_true, y_pred):

        # from Figure 1 in the paper
        y_U = {}
        y_L = {} # L(x)
        y_v = {} # v(x)
        y_T = {}  # y(x)
        for x in range(action_size):
            y_U[x] = y_pred[:, x]
            y_L[x] = y_pred[:, action_size + x]
            y_v[x] = y_pred[:, 2 * action_size + x]
            y_T[x] = y_true[:, x]
        
        N_ = {i:tf.cast(tf.size(y_T[i]), tf.float32) for i in range(action_size)}  # batch size

        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)

        # k_soft uses sigmoid
        k_soft = {i:tf.multiply(tf.sigmoid((y_U[i] - y_T[i]) * soften),
                             tf.sigmoid((y_T[i] - y_L[i]) * soften)) for i in range(action_size)}
        
    
        # k_hard uses sign step function
        k_hard = {i:tf.multiply(tf.maximum(0., tf.sign(y_U[i] - y_T[i])),
                             tf.maximum(0., tf.sign(y_T[i] - y_L[i]))) for i in range(action_size)}
       
        # MPIW_capt from equation 4
        MPIW_capt = {i:tf.divide(tf.reduce_sum(tf.abs(y_U[i] - y_L[i]) * k_hard[i]),
                              tf.reduce_sum(k_hard[i]) + 0.001) for i in range(action_size)}
        
 
        # equation 1 where k is k_soft
        PICP_soft = {i:tf.reduce_mean(k_soft[i]) for i in range(action_size)}
        

        # pi loss from section 4.2
        pi_loss = {i:MPIW_capt[i] + lambda_ * tf.sqrt(N_[i]) * tf.square(tf.maximum(0., 1. - alpha_ - PICP_soft[i])) for i in range(action_size)}
       
        y_piven = {i:y_v[i] * y_U[i] + (1 - y_v[i]) * y_L[i] for i in range(action_size)}  # equation 3
        y_piven = {i:tf.reshape(y_piven[i], (-1, 1)) for i in range(action_size)}
        y_T = {i:tf.reshape(y_T[i], (-1, 1)) for i in range(action_size)}
        v_loss = {i:tf.losses.mean_squared_error(y_T[i], y_piven[i]) for i in range(action_size)}# equation 5

        # piven_loss_ = (tf.sign(v_loss)) *(beta * pi_loss + (1 - beta) * v_loss)  # equation 6
        piven_loss_ = {i:(tf.sign(v_loss[i])) *(beta * pi_loss[i] + (1 - beta) * v_loss[i]) for i in range(action_size)}  # equation 6
        # piven_loss_ =  v_loss # equation 6
        loss_final = tf.concat([piven_loss_[i] for i in range(action_size)],axis=0)
        # print(v_loss)
        return loss_final

    return v_loss

def pi_loss(lambda_in=15., soften=160., alpha=0.05, beta=1. ,action_size=3):
    """

    :param lambda_in: lambda parameter
    :param soften: soften parameter
    :param alpha: confidence level (1-alpha)
    :param beta: balance parameter
    """

    def pi_loss(y_true, y_pred):

        y_U = {}
        y_L = {} # L(x)
        y_v = {} # v(x)
        y_T = {}  # y(x)
        for x in range(action_size):
            y_U[x] = y_pred[:, x]
            y_L[x] = y_pred[:, action_size + x]
            y_v[x] = y_pred[:, 2 * action_size + x]
            y_T[x] = y_true[:, x]
        
        N_ = {i:tf.cast(tf.size(y_T[i]), tf.float32) for i in range(action_size)}  # batch size

        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)

        # k_soft uses sigmoid
        k_soft = {i:tf.multiply(tf.sigmoid((y_U[i] - y_T[i]) * soften),
                             tf.sigmoid((y_T[i] - y_L[i]) * soften)) for i in range(action_size)}
        
    
        # k_hard uses sign step function
        k_hard = {i:tf.multiply(tf.maximum(0., tf.sign(y_U[i] - y_T[i])),
                             tf.maximum(0., tf.sign(y_T[i] - y_L[i]))) for i in range(action_size)}
       
        # MPIW_capt from equation 4
        MPIW_capt = {i:tf.divide(tf.reduce_sum(tf.abs(y_U[i] - y_L[i]) * k_hard[i]),
                              tf.reduce_sum(k_hard[i]) + 0.001) for i in range(action_size)}
        
 
        # equation 1 where k is k_soft
        PICP_soft = {i:tf.reduce_mean(k_soft[i]) for i in range(action_size)}
        

        # pi loss from section 4.2
        pi_loss = {i:MPIW_capt[i] + lambda_ * tf.sqrt(N_[i]) * tf.square(tf.maximum(0., 1. - alpha_ - PICP_soft[i])) for i in range(action_size)}
       
        y_piven = {i:y_v[i] * y_U[i] + (1 - y_v[i]) * y_L[i] for i in range(action_size)}  # equation 3
        y_piven = {i:tf.reshape(y_piven[i], (-1, 1)) for i in range(action_size)}
        y_T = {i:tf.reshape(y_T[i], (-1, 1)) for i in range(action_size)}
        v_loss = {i:tf.losses.mean_squared_error(y_T[i], y_piven[i]) for i in range(action_size)}# equation 5

        # piven_loss_ = (tf.sign(v_loss)) *(beta * pi_loss + (1 - beta) * v_loss)  # equation 6
        piven_loss_ = {i:(tf.sign(v_loss[i])) *(beta * pi_loss[i] + (1 - beta) * v_loss[i]) for i in range(action_size)}  # equation 6
        # piven_loss_ =  v_loss # equation 6
        loss_final = tf.concat([piven_loss_[i] for i in range(action_size)],axis=0)
        # print(v_loss)
        return loss_final

    return pi_loss

def piven_dqn_loss(lambda_in=15., soften=160., alpha=0.05, beta=0.5 ,action_size=3):
    """

    :param lambda_in: lambda parameter
    :param soften: soften parameter
    :param alpha: confidence level (1-alpha)
    :param beta: balance parameter
    """

    def piven_loss(y_true, y_pred):

        y_U = {}
        y_L = {} # L(x)
        y_v = {} # v(x)
        y_T = {}  # y(x)
        for x in range(action_size):
            y_U[x] = y_pred[:, x]
            y_L[x] = y_pred[:, action_size + x]
            y_v[x] = y_pred[:, 2 * action_size + x]
            y_T[x] = y_true[:, x]
        
        N_ = {i:tf.cast(tf.size(y_T[i]), tf.float32) for i in range(action_size)}  # batch size

        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)

        # k_soft uses sigmoid
        k_soft = {i:tf.multiply(tf.sigmoid((y_U[i] - y_T[i]) * soften),
                             tf.sigmoid((y_T[i] - y_L[i]) * soften)) for i in range(action_size)}
        
    
        # k_hard uses sign step function
        k_hard = {i:tf.multiply(tf.maximum(0., tf.sign(y_U[i] - y_T[i])),
                             tf.maximum(0., tf.sign(y_T[i] - y_L[i]))) for i in range(action_size)}
       
        # MPIW_capt 
        MPIW_capt = {i:tf.divide(tf.reduce_sum(tf.abs(y_U[i] - y_L[i]) * k_hard[i]),
                              tf.reduce_sum(k_hard[i]) + 0.001) for i in range(action_size)}
        
 
        # equation 1 where k is k_soft
        PICP_soft = {i:tf.reduce_mean(k_soft[i]) for i in range(action_size)}
        

        # pi loss 
        pi_loss = {i:MPIW_capt[i] + lambda_ * tf.sqrt(N_[i]) * tf.square(tf.maximum(0., 1. - alpha_ - PICP_soft[i])) for i in range(action_size)}
       
        y_piven = {i:y_v[i] * y_U[i] + (1 - y_v[i]) * y_L[i] for i in range(action_size)}  
        y_piven = {i:tf.reshape(y_piven[i], (-1, 1)) for i in range(action_size)}
        y_T = {i:tf.reshape(y_T[i], (-1, 1)) for i in range(action_size)}
        v_loss = {i:tf.losses.mean_squared_error(y_T[i], y_piven[i]) for i in range(action_size)}

        # piven_loss_ = (tf.sign(v_loss)) *(beta * pi_loss + (1 - beta) * v_loss) 
        piven_loss_ = {i:(tf.sign(v_loss[i])) *(beta * pi_loss[i] + (1 - beta) * v_loss[i]) for i in range(action_size)}  
        # piven_loss_ =  v_loss 
        loss_final = tf.concat([piven_loss_[i] for i in range(action_size)],axis=0)
        # print(v_loss)
        return loss_final

    return piven_loss

"""
    Double DQN Agent for the Cartpole
    it uses Neural Network to approximate q function
    and replay memory & target q network
"""
class DoubleDQNAgent:
    def __init__(self, state_size, action_size,windows,isPiven,policy="point_prediction",lr = 0.0001,gama=0.99,piven_beta=0.5,piven_alpha=0.25,bias_init=np.append(np.repeat(2.,2) ,np.repeat(-2.,2)),load_model=False,load_name=None,load_path='',load_episode=-1,seed=2,swap=False,clip = False,test=False,largev=False,env_name='cartpole',pretrained=False):
        """
        Agent for the IPIV-RL
        :param state_size: state size
        :param action_size: action size
        :param windows:
        :param isPiven: boolean for ipiv
        :param policy: policy for the agent
        :param lr: learning rate
        :param gama: gama hyperparameter of DQN
        :param piven_beta: IPIV alpha
        :param piven_alpha: IPIV beta
        :param bias_init: inital bias for the networks PIs
        :param load_model: boolean for loading model
        :param load_name: model name
        :param load_path: path to the model
        :param load_episode: episode number
        :param seed: seed for the random
        :param swap: boolean for swap
        :param clip: boolean for gradient clipping
        :param env_name: environment name
        :param pretrained: boolean for pretrained model
        """
        if not test:
        # WandB init
        #     self.run = wandb.init(project='test-proj',reinit=True) #CartpolePtoject
            # self.run = wandb.init(project='LunarLender',reinit=True) #LunarLender Project
            self.run = wandb.init(project='Fixes',reinit=True) #AtariBreakout Project
            self.run_name = wandb.run.name
            # wandb.run.save()
            #Logging to wandb
            self.config = wandb.config
            self.config.envName = env_name
            self.config.is_piven = isPiven
            self.config.learning_rate= lr
            self.config.seed = seed
            self.config.policy = policy
            self.config.envName = env_name
            if isPiven:
                self.config.swap=swap
                self.config.clipping = clip
                self.config.largev = largev
                self.config.dqn_gama=gama
                self.config.pi_beta=piven_beta
                self.config.pi_alpha=piven_alpha
                self.config.bias_init = bias_init
                self.config.n_windows = windows
            
        
        self.seed = seed
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        random.seed(seed)
        
        self.env_name = env_name
        self.render = False
        self.load_model = load_model
        self.largev = largev
        # get size of state and action
        self.add_features = 3
        self.windows = windows
        self.extra_f  = self.windows*self.add_features*action_size
        if isPiven:
            if env_name == 'Atari':
                self.state_size = state_size
            else:
                self.state_size = state_size+self.extra_f
        else:
            self.state_size =state_size
        self.action_size = action_size
        self.isPiven=isPiven
        self.swap=swap
        self.clip = clip
        # these is hyper parameters for the Double DQN
        self.discount_factor = gama
        self.learning_rate = lr
        self.epsilon = .995
        self.epsilon_max = .995
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        # self.batch_size = 64 # Cartpole
        self.batch_size = 32
        if isPiven:
            if env_name == 'Atari':
                self.train_start = 10000
            else:
                self.train_start = 128
        else:
            self.train_start = 1000
        self.piven_beta = piven_beta
        self.piven_alpha = piven_alpha
        self.policy=policy
        # create replay memory using deque
        # self.memory = deque(maxlen=2000) # Cartpole
        # self.memory = deque(maxlen=5000) # LunarLender
        self.memory = deque(maxlen=1000000) # Atari

        # create main model and target model
        if isPiven:
            if env_name == 'Atari':
                self.model = self.build_model_atari(bias_init=bias_init)
                self.target_model = self.build_model_atari(bias_init=bias_init)
            else:
                if pretrained:
                    tp_path = "save_model/LunarLander/NoPiven/dqn_LunarLander-v2_weights_seed1238.h5"
                    self.model = self.build_pt(bias_init = bias_init,tp_path=tp_path)
                    self.target_model = self.build_pt(bias_init = bias_init,tp_path=tp_path)
                    
                self.model = self.build_piven(bias_init=bias_init)
                self.target_model = self.build_piven(bias_init=bias_init)
        else:    
            self.model = self.build_model()
            self.target_model = self.build_model()
        self.model.summary()
        # initialize target model
        self.update_target_model()

        if self.load_model:
            root = './'
            # root = 'new_attemptJuly2021/'
            # self.model.load_weights(f"./save_model/cartpole_ddqn_{load_name}_{load_episode}.h5")
            # self.model.load_weights(f"./save_model/{load_name}.h5")
            # self.model.load_weights(f"{root}fine-tune/{load_name}.h5")
            self.model.load_weights(load_path)
    def build_model_atari(self,bias_init):
        """
        Builds the Q-network model.
        """
        # Neural Net for Deep-Q learning Model
        INPUT_SHAPE = (84, 84)
        WINDOW_LENGTH = 4
        # env = gym.make('BreakoutDeterministic-v4')
        # np.random.seed(123)
        # env.seed(123)
        nb_actions = self.action_size
        input_shape = self.state_size
        # model = Sequential()
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
        opt = tf.keras.optimizers.Adam(lr=self.learning_rate,clipvalue=.6,clipnorm=1.)
        lossf = piven_dqn_loss(beta=self.piven_beta,lambda_in=15.0,alpha=self.piven_alpha,action_size=nb_actions)
        # lossf = piven_dqn_loss_normal(beta=self.piven_beta,lambda_in=15.0,alpha=.25,action_size=self.action_size)
        pi_lossf = pi_loss(beta=1.,lambda_in=15.0,alpha=.25,action_size=nb_actions)
        v_lossf = v_loss(beta=0.,lambda_in=15.0,alpha=.25,action_size=nb_actions)
        model.compile(loss=lossf,metrics=[pi_lossf,v_lossf],optimizer=opt)
        return model

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer=he_uniform(seed=self.seed),name='h1'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer=he_uniform(seed=self.seed),name='h2'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer=he_uniform(seed=self.seed),name='out'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def build_piven(self,bias_init):
        b_norm=True

        inputs = Input(shape=(self.state_size,),name='State_input')
        X = Dense(64, input_shape=(self.state_size,), activation="relu", kernel_initializer=he_uniform(seed=self.seed), name='h1')(inputs)
        if b_norm:
            X = BatchNormalization(name='Bnorm1')(X)
        X = Dense(64, activation="relu", kernel_initializer=he_uniform(seed=self.seed), name='h2')(X)
        if b_norm:
            X = BatchNormalization(name='Bnorm2')(X)
        X = Dense(64, activation="relu", kernel_initializer=he_uniform(seed=self.seed), name='h3')(X)
        if b_norm:
            X = BatchNormalization(name='Bnorm3')(X)
        X = Dense(64, activation="relu", kernel_initializer=he_uniform(seed=self.seed), name='h4')(X)
        if b_norm:
            X = BatchNormalization(name='Bnorm4')(X)
        X = Dense(32, activation="relu", kernel_initializer=he_uniform(seed=self.seed), name='h5')(X)
        if b_norm:
            X = BatchNormalization(name='Bnorm5')(X)


        pi = Dense(2*self.action_size, activation='linear',kernel_initializer=RandomNormal(mean=0.0, stddev=0.2,seed=self.seed),
                                        bias_initializer=Constant(value=bias_init), name='pi')(X)# pi initialization using bias
        activation = "linear" if self.largev else "sigmoid"
        q = Dense(self.action_size, activation=activation, kernel_initializer=he_uniform(seed=self.seed), name='q')(X)

        piven_out = Concatenate(name='piven_out')([pi, q])
        model = Model(inputs=inputs, outputs=[piven_out], name='piven_model')
        # compile
        if not self.clip:
            opt = tf.keras.optimizers.Adam(lr=self.learning_rate)
        else:
            opt = tf.keras.optimizers.Adam(lr=self.learning_rate,clipvalue=.6,clipnorm=1.)
        lossf = piven_dqn_loss(beta=self.piven_beta,lambda_in=15.0,alpha=self.piven_alpha,action_size=self.action_size)
        pi_lossf = pi_loss(beta=1.,lambda_in=15.0,alpha=.25,action_size=self.action_size)
        v_lossf = v_loss(beta=0.,lambda_in=15.0,alpha=.25,action_size=self.action_size)
        model.compile(loss=lossf,metrics=[pi_lossf,v_lossf],optimizer=opt)
        model.summary()
        
        # 13.9 New idea train_swap
        if self.swap:
            for layer in model.layers:
                if layer.name=='pi' or layer.name=='q':
                    layer.trainable=False
                    
        return model
    
    def build_pt(self,bias_init,tp_path):
        """
        Use pretrained lunarlander network as basis for IPIV-DQN
        """
        pretraind_path = tp_path
        pretraind_model = keras.models.load_model(pretraind_path, custom_objects={'piven_loss': piven_dqn_loss(action_size=self.action_size)})
        self.model = keras.models.clone_model(pretraind_model)
        #replace the last layer with a new one
        self.model.layers.pop()
        print(self.model.get_weights())
        #add new dense layer
        # self.model.layers.add(Dense(self.action_size*2, activation='softmax', kernel_initializer=bias_init))
        inputs = Input(shape=(self.state_size,), name='State_input')
        X = self.model.layers[1](inputs)
        for layer in self.model.layers[2:]:
            X = layer(X)

        activation = "linear" if self.largev else "sigmoid"
        pi = Dense(2*self.action_size, activation='linear',kernel_initializer=tf.keras.initializers.Zeros(),
                                        bias_initializer=Constant(value=bias_init), name='pi')(X) # pi initialization using bias
        q = Dense(self.action_size, activation="sigmoid", kernel_initializer='he_uniform', name='q')(X)
        piven_out = Concatenate(name='piven_out')([pi, q])
        model = Model(inputs=inputs, outputs=[piven_out], name='piven_model')
        # compile
        if not self.clip:
            opt = tf.keras.optimizers.Adam(lr=self.learning_rate)
        else:
            opt = tf.keras.optimizers.Adam(lr=self.learning_rate,clipvalue=.6,clipnorm=1.)
        lossf = piven_dqn_loss(beta=self.piven_beta,lambda_in=15.0,alpha=self.piven_alpha,action_size=self.action_size)
        # lossf = piven_dqn_loss_normal(beta=self.piven_beta,lambda_in=15.0,alpha=.25,action_size=self.action_size)
        pi_lossf = pi_loss(beta=1.,lambda_in=15.0,alpha=.25,action_size=self.action_size)
        v_lossf = v_loss(beta=0.,lambda_in=15.0,alpha=.25,action_size=self.action_size)
        model.compile(loss=lossf,metrics=[pi_lossf,v_lossf],optimizer=opt)
        model.summary()
        return model
    
    # after some time interval update the target model to be same with model
    def swap_trainable(self):
        for layer in self.model.layers:
            layer.trainable=not layer.trainable
        
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # The function takes PI-DQN prediction and returns Q-values.
    def predict_to_q(self,preds):
        ans = preds[:, 0] * preds[:, 2 * self.action_size] + (1 - preds[:, 2 * self.action_size]) * preds[:,
                                                                                                    self.action_size]
        ans = np.reshape(ans, (-1, 1))
        for x in range(1, self.action_size):
            append = preds[:, 0 + x] * preds[:, 2 * self.action_size + x] + (
                        1 - preds[:, 2 * self.action_size + x]) * preds[:,
                                                                self.action_size + x]
            append = np.reshape(append, [-1, 1])
            ans = np.concatenate([ans, append], 1)
        return ans

    def _choose_action(self,predict,q_val,alpha=0.9):

        if  self.policy == 'point_prediction':
            return np.argmax(q_val)
        
        elif self.policy == 'alpha_method':
            bounds_len =np.array( [predict[0][x]-predict[0][x+self.action_size] for x in range(self.action_size)])
            values = alpha*q_val + (1-alpha)*(1./bounds_len)
            return np.argmax(values)
    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size),None,None
        else:
            # state = process_state_batch(state.reshape((1,) + state.shape))
            q_value = self.model.predict(state)
            if self.isPiven:
                full_pred = q_value
                q_value = self.predict_to_q(q_value)
                action = self._choose_action(full_pred,q_value)
                return action,q_value,full_pred
            return np.argmax(q_value[0]),q_value,None
        
    def test_action(self,state):
        q_value = self.model.predict(state)
#         print(q_value)
        if self.isPiven:
            q_value = self.predict_to_q(q_value)
        return np.argmax(q_value[0])
        

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

            
    def get_up_low(self,hist):
        hist = [x[0] for x in hist]
        uppers = []
        lowers = []
        for x in range(self.action_size):
            uppers.append([h[x] for h in hist])
            lowers.append([h[x+self.action_size] for h in hist])
        uppers = np.array(uppers)
        lowers = np.array(lowers)
        return uppers,lowers

    def intersect(self,up,low,lens):
        intervals = []
        act1I = np.array([])
        act2I = np.array([])
        for act in range(self.action_size):
            intervals.append([(x,y,z) for x,y,z in zip(up[act],low[act],lens[act])])
        for (act1,act2) in zip(*intervals):
            intersect = max(0,min(act1[0],act2[0]) - max(act1[1],act2[1]))
            act1I = np.append(act1I,intersect/act1[2])
            act2I = np.append(act2I,intersect/act2[2])
    #         print(act1I)
        return[act1I.mean(),act2I.mean()]

    def calc_extras(self,steps,ns):
        # print(len(steps))
        res =np.array([])
        for n in ns:
            if len(steps)<n:
                # print(n)
                ans = np.zeros(self.action_size*self.add_features)
            else:
                hist = np.array(steps[-n:])
                uppers,lowers = self.get_up_low(hist)
                lens = []
                for act in range(self.action_size):
                    lens.append(uppers[act]-lowers[act])
                lens = np.array(lens)
                mean = lens.mean(axis=1)/100 #Interval size
                std = lens.std(axis=1)#Interval std
                ans = np.append(mean,std)
                ans = np.append(ans,self.intersect(uppers,lowers,lens))#Intersection precentage for both intervals
            res = np.append(res,ans)
        return res

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        delta = 0.01
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)
        if self.env_name == 'Atari':
            update_input = np.zeros((batch_size,)+ self.state_size)
            update_target = np.zeros((batch_size,) + self.state_size)
        else:   
            update_input = np.zeros((batch_size, self.state_size))
            update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])
            
        #TODO : CONV Adaptation: insert call for proccess state batch
        if self.env_name == 'Atari':
            update_input = process_state_batch(update_input)
            update_target = process_state_batch(update_target)
        pred = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)
        if self.isPiven:
            target_lb = [x[self.action_size:self.action_size*2] for x in pred]
            target = self.predict_to_q(pred)
            target_next = self.predict_to_q(target_next)
            target_val = self.predict_to_q(target_val)
        else:
            target=pred

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                if self.isPiven:
                    reward[i] = target_lb[i][action[i]] +delta if reward[i]==-100 else reward[i] # Keep reward inside the interval
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])


        loss = self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0,callbacks=[WandbCallback()])

        return loss