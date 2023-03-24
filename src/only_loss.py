import tensorflow as tf
import numpy as np
def piven_dqn_loss(lambda_in=15., soften=160., alpha=0.05, beta=0.5 ,action_size=3):
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
        print(N_)
        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)

        # k_soft uses sigmoid
        k_soft = {i:tf.multiply(tf.sigmoid((y_U[i] - y_T[i]) * soften),
                             tf.sigmoid((y_T[i] - y_L[i]) * soften)) for i in range(action_size)}
        
        print(k_soft)
        # k_hard uses sign step function
        k_hard = {i:tf.multiply(tf.maximum(0., tf.sign(y_U[i] - y_T[i])),
                             tf.maximum(0., tf.sign(y_T[i] - y_L[i]))) for i in range(action_size)}
        print(k_hard)
        # MPIW_capt from equation 4
        MPIW_capt = {i:tf.divide(tf.reduce_sum(tf.abs(y_U[i] - y_L[i]) * k_hard[i]),
                              tf.reduce_sum(k_hard[i]) + 0.001) for i in range(action_size)}
        
        print(MPIW_capt)
        # equation 1 where k is k_soft
        PICP_soft = {i:tf.reduce_mean(k_soft[i]) for i in range(action_size)}
        
        print(PICP_soft)
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

    return piven_loss

def piven_loss(lambda_in=15., soften=160., alpha=0.05, beta=0.5):
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
        y_U = y_pred[:, 0] # U(x)
        y_L = y_pred[:, 1] # L(x)
        y_v = y_pred[:, 2] # v(x)
        y_T = y_true[:, 0] # y(x)

        N_ = tf.cast(tf.size(y_T), tf.float32)  # batch size
        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)

        # k_soft uses sigmoid
        k_soft = tf.multiply(tf.sigmoid((y_U - y_T) * soften), 
                             tf.sigmoid((y_T - y_L) * soften))

        # k_hard uses sign step function
        k_hard = tf.multiply(tf.maximum(0., tf.sign(y_U - y_T)), 
                             tf.maximum(0., tf.sign(y_T - y_L)))

        # MPIW_capt from equation 4
        MPIW_capt = tf.divide(tf.reduce_sum(tf.abs(y_U - y_L) * k_hard),
                              tf.reduce_sum(k_hard) + 0.001)

        # equation 1 where k is k_soft
        PICP_soft = tf.reduce_mean(k_soft) 

        # pi loss from section 4.2
        pi_loss =  MPIW_capt  + lambda_ * tf.sqrt(N_) * tf.square(tf.maximum(0., 1. - alpha_ - PICP_soft))

        y_piven = y_v * y_U + (1 - y_v) * y_L # equation 3 
        y_piven = tf.reshape(y_piven, (-1, 1))

        v_loss = tf.losses.mean_squared_error(y_true, y_piven)  # equation 5
        piven_loss_ = beta * pi_loss + (1-beta) * v_loss # equation 6

        return piven_loss_

    return piven_loss

loss2 = piven_dqn_loss(beta=.5,lambda_in=15.0,alpha=.25,action_size=2)
loss1 = piven_loss()

dummy = np.array([[24.,24.,5.,6.,0.4,0.5]])
y_true_d = np.array([[35.,15.]])
loss2(y_true_d,dummy)

d1 = np.array([[4,0.5,0.5]])
dt = np.array([[2.]])
loss1(dt,d1)
