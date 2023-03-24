import numpy as np
import tensorflow as tf

"""
Author: Amir Loewenthal

This File contains the implementation of the loss function for the IPIV-DQN model.
"""

def v_loss(lambda_in=15., soften=160., alpha=0.05, beta=0., action_size=3):
    """

    :param lambda_in: lambda parameter
    :param soften: soften parameter
    :param alpha: confidence level (1-alpha)
    :param beta: balance parameter
    """

    def v_loss(y_true, y_pred):
        y_U = {}
        y_L = {}  # L(x)
        y_v = {}  # v(x)
        y_T = {}  # y(x)
        for x in range(action_size):
            y_U[x] = y_pred[:, x]
            y_L[x] = y_pred[:, action_size + x]
            y_v[x] = y_pred[:, 2 * action_size + x]
            y_T[x] = y_true[:, x]

        N_ = {i: tf.cast(tf.size(y_T[i]), tf.float32) for i in range(action_size)}  # batch size

        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)

        # k_soft uses sigmoid
        k_soft = {i: tf.multiply(tf.sigmoid((y_U[i] - y_T[i]) * soften),
                                 tf.sigmoid((y_T[i] - y_L[i]) * soften)) for i in range(action_size)}

        # k_hard uses sign step function
        k_hard = {i: tf.multiply(tf.maximum(0., tf.sign(y_U[i] - y_T[i])),
                                 tf.maximum(0., tf.sign(y_T[i] - y_L[i]))) for i in range(action_size)}

        # MPIW_capt from equation 1
        MPIW_capt = {i: tf.divide(tf.reduce_sum(tf.abs(y_U[i] - y_L[i]) * k_hard[i]),
                                  tf.reduce_sum(k_hard[i]) + 0.001) for i in range(action_size)}

        PICP_soft = {i: tf.reduce_mean(k_soft[i]) for i in range(action_size)}

        # pi loss from section 4.1.2
        pi_loss = {i: MPIW_capt[i] + lambda_ * tf.sqrt(N_[i]) * tf.square(tf.maximum(0., 1. - alpha_ - PICP_soft[i]))
                   for i in range(action_size)}

        y_piven = {i: y_v[i] * y_U[i] + (1 - y_v[i]) * y_L[i] for i in range(action_size)}
        y_piven = {i: tf.reshape(y_piven[i], (-1, 1)) for i in range(action_size)}
        y_T = {i: tf.reshape(y_T[i], (-1, 1)) for i in range(action_size)}
        v_loss = {i: tf.losses.mean_squared_error(y_T[i], y_piven[i]) for i in range(action_size)}

        piven_loss_ = {i: (tf.sign(v_loss[i])) * (beta * pi_loss[i] + (1 - beta) * v_loss[i]) for i in
                       range(action_size)}  # equation 4

        loss_final = tf.concat([piven_loss_[i] for i in range(action_size)], axis=0)
        # print(v_loss)
        return loss_final

    return v_loss


def pi_loss(lambda_in=15., soften=160., alpha=0.05, beta=1., action_size=3):
    """

    :param lambda_in: lambda parameter
    :param soften: soften parameter
    :param alpha: confidence level (1-alpha)
    :param beta: balance parameter
    """

    def pi_loss(y_true, y_pred):
        y_U = {}
        y_L = {}  # L(x)
        y_v = {}  # v(x)
        y_T = {}  # y(x)
        for x in range(action_size):
            y_U[x] = y_pred[:, x]
            y_L[x] = y_pred[:, action_size + x]
            y_v[x] = y_pred[:, 2 * action_size + x]
            y_T[x] = y_true[:, x]

        N_ = {i: tf.cast(tf.size(y_T[i]), tf.float32) for i in range(action_size)}  # batch size

        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)

        k_soft = {i: tf.multiply(tf.sigmoid((y_U[i] - y_T[i]) * soften),
                                 tf.sigmoid((y_T[i] - y_L[i]) * soften)) for i in range(action_size)}

        k_hard = {i: tf.multiply(tf.maximum(0., tf.sign(y_U[i] - y_T[i])),
                                 tf.maximum(0., tf.sign(y_T[i] - y_L[i]))) for i in range(action_size)}

        MPIW_capt = {i: tf.divide(tf.reduce_sum(tf.abs(y_U[i] - y_L[i]) * k_hard[i]),
                                  tf.reduce_sum(k_hard[i]) + 0.001) for i in range(action_size)}

        PICP_soft = {i: tf.reduce_mean(k_soft[i]) for i in range(action_size)}

        pi_loss = {i: MPIW_capt[i] + lambda_ * tf.sqrt(N_[i]) * tf.square(tf.maximum(0., 1. - alpha_ - PICP_soft[i]))
                   for i in range(action_size)}

        y_piven = {i: y_v[i] * y_U[i] + (1 - y_v[i]) * y_L[i] for i in range(action_size)}
        y_piven = {i: tf.reshape(y_piven[i], (-1, 1)) for i in range(action_size)}
        y_T = {i: tf.reshape(y_T[i], (-1, 1)) for i in range(action_size)}
        v_loss = {i: tf.losses.mean_squared_error(y_T[i], y_piven[i]) for i in range(action_size)}

        piven_loss_ = {i: (tf.sign(v_loss[i])) * (beta * pi_loss[i] + (1 - beta) * v_loss[i]) for i in
                       range(action_size)}

        loss_final = tf.concat([piven_loss_[i] for i in range(action_size)], axis=0)

        return loss_final

    return pi_loss


def ipiv_dqn_loss(lambda_in=15., soften=160., alpha=0.05, beta=0.5, action_size=3):
    """

    :param lambda_in: lambda parameter
    :param soften: soften parameter
    :param alpha: confidence level (1-alpha)
    :param beta: balance parameter
    """

    def ipiv_loss(y_true, y_pred):
        y_U = {}
        y_L = {}  # L(x)
        y_v = {}  # v(x)
        y_T = {}  # y(x)
        for x in range(action_size):
            y_U[x] = y_pred[:, x]
            y_L[x] = y_pred[:, action_size + x]
            y_v[x] = y_pred[:, 2 * action_size + x]
            y_T[x] = y_true[:, x]

        N_ = {i: tf.cast(tf.size(y_T[i]), tf.float32) for i in range(action_size)}  # batch size

        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)

        # k_soft uses sigmoid
        k_soft = {i: tf.multiply(tf.sigmoid((y_U[i] - y_T[i]) * soften),
                                 tf.sigmoid((y_T[i] - y_L[i]) * soften)) for i in range(action_size)}

        # k_hard uses sign step function
        k_hard = {i: tf.multiply(tf.maximum(0., tf.sign(y_U[i] - y_T[i])),
                                 tf.maximum(0., tf.sign(y_T[i] - y_L[i]))) for i in range(action_size)}

        # MPIW_capt
        MPIW_capt = {i: tf.divide(tf.reduce_sum(tf.abs(y_U[i] - y_L[i]) * k_hard[i]),
                                  tf.reduce_sum(k_hard[i]) + 0.001) for i in range(action_size)}

        PICP_soft = {i: tf.reduce_mean(k_soft[i]) for i in range(action_size)}

        # pi loss
        pi_loss = {i: MPIW_capt[i] + lambda_ * tf.sqrt(N_[i]) * tf.square(tf.maximum(0., 1. - alpha_ - PICP_soft[i]))
                   for i in range(action_size)}

        y_ipiv = {i: y_v[i] * y_U[i] + (1 - y_v[i]) * y_L[i] for i in range(action_size)}
        y_ipiv = {i: tf.reshape(y_ipiv[i], (-1, 1)) for i in range(action_size)}
        y_T = {i: tf.reshape(y_T[i], (-1, 1)) for i in range(action_size)}
        v_loss = {i: tf.losses.mean_squared_error(y_T[i], y_ipiv[i]) for i in range(action_size)}

        ipiv_loss_ = {i: (tf.sign(v_loss[i])) * (beta * pi_loss[i] + (1 - beta) * v_loss[i]) for i in
                      range(action_size)}

        loss_final = tf.concat([ipiv_loss_[i] for i in range(action_size)], axis=0)

        return loss_final

    return ipiv_loss
