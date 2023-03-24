from collections import deque
from keras.layers import Dense,Input,BatchNormalization,Concatenate
from keras.optimizers import Adam
from keras.models import Sequential,Model
from keras.initializers import RandomNormal, Constant, GlorotNormal,he_uniform
import random
import tensorflow as tf
import numpy as np
import gym
from collections import deque
# piven loss definition
root = 'new_attemptJuly2021/'
def piven_loss_no_sign(lambda_in=15., soften=160., alpha=0.05, beta=0.5,action_size=3):
    """
    For easy understanding, I referred each equation in the implementation, 
    to the corresponding equation in the article.

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
        MPIW_capt = {i:tf.divide(tf.reduce_sum(tf.maximum(y_U[i] - y_L[i],0) * k_hard[i]),
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
        piven_loss_ = {i: (beta * pi_loss[i] + (1 - beta) * v_loss[i]) for i in range(action_size)}  # equation 6
        # piven_loss_ =  v_loss # equation 6
        loss_final = tf.concat([piven_loss_[i] for i in range(action_size)],axis=0)
        # print(v_loss)
        return loss_final

    return piven_loss

def _gen_uniform_noise(input,sigma=0.2):
    noise = np.random.uniform(input-sigma,input+sigma)
    return noise

def dqn_model(state_size=4,action_size=2,lr=0.01):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(24, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(action_size, activation='linear',
                    kernel_initializer='he_uniform'))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    model.load_weights(f"{root}save_model/NoPiven_model/cartpole_ddqn_NoPiven_100.h5")
    return model

def build_piven(bias_init,state_size=4,action_size=2,piven_beta=0.5,lr=0.001):
    b_norm=True

    inputs = Input(shape=(state_size,))
    X = Dense(64, input_shape=(state_size,), activation="relu", kernel_initializer='he_uniform', name='h1')(inputs)
    if b_norm:
        X = BatchNormalization()(X)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform', name='h2')(X)
    if b_norm:
        X = BatchNormalization()(X)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform', name='h3')(X)
    if b_norm:
        X = BatchNormalization()(X)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform', name='h4')(X)
    if b_norm:
        X = BatchNormalization()(X)
    X = Dense(32, activation="relu", kernel_initializer='he_uniform', name='h5')(X)
    if b_norm:
        X = BatchNormalization()(X)

    pi = Dense(2*action_size, activation='linear',kernel_initializer=RandomNormal(mean=0.0, stddev=0.2),
                                    bias_initializer=Constant(value=bias_init), name='pi')(X) # pi initialization using bias
    q = Dense(action_size, activation="sigmoid", kernel_initializer='he_uniform', name='q')(X)

    piven_out = Concatenate(name='piven_out')([pi, q])
    model = Model(inputs=inputs, outputs=[piven_out], name='piven_model')
    # compile
    opt = tf.keras.optimizers.Adam(lr=lr)
    lossf = piven_loss_no_sign(beta=piven_beta,lambda_in=15.0,alpha=.05,action_size=action_size)
    # lossf = piven_loss_no_sign(beta=self.piven_beta,lambda_in=15.0,alpha=.25,action_size=action_size)
    # pi_lossf = pi_loss(beta=1.,lambda_in=15.0,alpha=.25,action_size=action_size)
    # v_lossf = v_loss(beta=0.,lambda_in=15.0,alpha=.25,action_size=action_size)
    model.compile(loss=lossf,optimizer=opt)
    model.summary()
    return model

def predict_to_q(preds,action_size=2):
    ans = preds[:, 0] * preds[:, 2 * action_size] + (1 - preds[:, 2 * action_size]) * preds[:,
                                                                                                action_size]
    ans = np.reshape(ans, (-1, 1))
    for x in range(1, action_size):
        append = preds[:, 0 + x] * preds[:, 2 * action_size + x] + (
                    1 - preds[:, 2 * action_size + x]) * preds[:,
                                                            action_size + x]
        append = np.reshape(append, [-1, 1])
        ans = np.concatenate([ans, append], 1)
    return ans

def shuffle_add_noise(X,Y,n_splits=2):
    """
    Split data to equal parts. add uniformed noise in different magnitudes to each part.
    
    """
    x_parts = np.array_split(X,n_splits)
    y_parts = np.array_split(Y,n_splits)
    x_tupples =[(i,x_parts[i]) for i in range(n_splits)]
    y_tupples =[(i,y_parts[i]) for i in range(n_splits)]
    np.random.shuffle(x_tupples)
    noised = []
    for i in range(n_splits):
        sigma = i/10
        noised.append((x_tupples[i][0],_gen_uniform_noise(x_tupples[i][1],sigma)))
    X = np.concatenate([t[1] for t in noised])
    Y = np.concatenate([y_tupples[x_t[0]][1] for x_t in x_tupples])
    return X,Y

def train_pi(train_model,memory,iter):
    # batch_size = min(64, len(memory))
    # mini_batch = random.sample(memory, batch_size)
    X = np.array([memory[x][0] for x in range(len(memory))])
    X = np.reshape(X,(X.shape[0],X.shape[2]))
    Y = np.array([memory[x][1] for x in range(len(memory))])
    Y = np.reshape(Y,(Y.shape[0],Y.shape[2]))
    # X,Y = shuffle_add_noise(X,Y,3)
    return train_model.fit(X,Y,batch_size=32,epochs=10,verbose=1)

def test_agent(train_model,env):
    scores = []
    for e in range(10):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1,4])

        while not done:

            # get action for the current state and go one step in environment
            q_value = train_model.predict(state)
            print(q_value)
            q_value =  predict_to_q(q_value)
            action = np.argmax(q_value[0])
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            score += reward
            state = next_state
        print(f"Done in {score} episodes.")
        scores.append(score)
    return np.array(scores).mean()
    
if __name__=='__main__':
    model_1 = dqn_model()
    bias_init=np.append(np.repeat(75.,2) ,np.repeat(40.,2)) # For no noise working
    # bias_init=np.append(np.repeat(75.,2) ,np.repeat(20.,2))
    memory = deque(maxlen=60000)
    train_model = build_piven(bias_init=bias_init)
    env = gym.make('CartPole-v1')
    loss=9999
    score=0
    iter=0
    while loss>.01 :
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 4])
        steps = list()
        X,Y=[],[]
        while not done:

            # get action for the current state and go one step in environment
            action = np.argmax(model_1(state))
            next_state, reward, done, info = env.step(action)
            steps.append(model_1(state))
            memory.append((state,model_1(state)))
            next_state = np.reshape(next_state, [1, 4])
            score += reward
            state = next_state
        hist = train_pi(train_model,memory,iter)
        iter+=1
        loss = hist.history['loss'][-1]
        # val_loss = hist.history['val_loss'][-1]
        score = test_agent(train_model,env)
        print(f"Score = {score}")
        if score>=100:
            train_model.save_weights(f'{root}fine-tune/ft_base_{score}_{iter}_{loss}.h5') 
        if score>=400:
           train_model.save_weights(f'{root}only_pi_score_n_{score}_{iter}.h5') 
    print(loss)
    train_model.save_weights(f'{root}only_pi.h5')
