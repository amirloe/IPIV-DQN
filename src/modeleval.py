from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import keras
import gym
import numpy as np
import tensorflow as tf
import tqdm as tqdm
import tqdm.notebook as tq
from keras.models import Model, load_model
from keras.layers import Dense, Concatenate, Input, BatchNormalization
from keras.initializers import RandomNormal, Constant, GlorotNormal, he_uniform
from keras.initializers import RandomNormal, Constant, GlorotNormal
from cartpole import CartPole,Breakout
from losses import ipiv_dqn_loss, pi_loss, v_loss
from samdpModels import model_setup
from convDQNModels import getconv_vanilla_model, getconv_ipiv_model
import torch
from processing import process_observation,process_state_batch
USE_CUDA = torch.cuda.is_available()

class CartpoleIPIVAgentsEval:
    """
    Evaluates and compares the performance of two agents.
    The first agent is a vanilla-DQN agent, the second is an IPIV-DQN agent.
    """

    def __init__(self, model1_path, model2_path,bu_model_path, name1, name2, is_ipiv=True,model1SA=False,atari=False):
        """

        :param model1_path: path to the first model
        :param model2_path: path to the second model
        :param name1: name of the first model
        :param name2: name of the second model
        :param is_ipiv: if true, the model is trained with IPIV, otherwise it is trained with vanilla DQN
        """
        self.seed = 6666
        self.atari = atari
        self.model1SA = model1SA
        
        if not self.atari:
            self.bias_init = np.append(np.repeat(75., 2), np.repeat(40., 2))
            # get size of state and action from environment
            self.env1 = CartPole()
            self.env2 = CartPole()
            self.state_size = self.env1.observation_space.shape[0]
            self.state_space = self.env1.observation_space.shape[0]
            self.action_size = self.env1.action_space.n
            
            if not model1SA:
                self.model1 = load_model(model1_path, custom_objects={'piven_loss': ipiv_dqn_loss(
                    action_size=self.action_size)})
            else:
                self.model1 = model_setup('CartPole-v1', self.env1, True, USE_CUDA, True, 1)
                self.model1.features.load_state_dict(torch.load(model1_path))
            self.model2 = self.build_ipiv(bias_init=self.bias_init)
            self.model2.load_weights(model2_path)
            self.bu_model = load_model(bu_model_path, custom_objects={'piven_loss': ipiv_dqn_loss(
                action_size=self.action_size)})
        else:
            self.bias_init = np.append(np.repeat(75., 4), np.repeat(40., 4))
            self.env1 = Breakout()
            self.env2 = Breakout()
            INPUT_SHAPE = (84, 84)
            WINDOW_LENGTH = 4
            self.input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
            self.state_size = self.env1.observation_space.shape[0]
            self.state_space = self.env1.observation_space.shape
            self.action_size = self.env1.action_space.n
            
            self.model1 = getconv_vanilla_model(self.input_shape, self.action_size)
            self.model1.load_weights(model1_path)

            self.model2 = getconv_ipiv_model(self.input_shape, self.action_size,self.bias_init)
            self.model2.load_weights(model2_path)
            
            self.bu_model = getconv_vanilla_model(self.input_shape, self.action_size)
            self.bu_model.load_weights(bu_model_path)        

        self.name1 = name1
        self.name2 = name2
        self.isPiven = is_ipiv

        self.log = {}
        self.reset_log()

    def build_ipiv(self, bias_init):
        """
        Builds the IPIV-DQN model.
        :param bias_init: bias initialization for the interval output layer
        :return: compiled model
        """
        b_norm = True

        inputs = Input(shape=(self.state_size,), name='State_input')
        X = Dense(64, input_shape=(self.state_size,), activation="relu", kernel_initializer=he_uniform(seed=self.seed),
                  name='h1')(inputs)
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

        pi = Dense(2 * self.action_size, activation='linear',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.2, seed=self.seed),
                   bias_initializer=Constant(value=bias_init), name='pi')(X)  # pi initialization using bias
        q = Dense(self.action_size, activation="sigmoid", kernel_initializer=he_uniform(seed=self.seed), name='q')(X)
        ipiv_out = Concatenate(name='ipiv_out')([pi, q])
        model = Model(inputs=inputs, outputs=[ipiv_out], name='piven_model')
        # compile
        opt = tf.keras.optimizers.Adam(lr=1e-4)

        lossf = ipiv_dqn_loss(lambda_in=15.0, alpha=0.15, beta=0.9, action_size=self.action_size)
        pi_lossf = pi_loss(beta=1., lambda_in=15.0, alpha=.25, action_size=self.action_size)
        v_lossf = v_loss(beta=0., lambda_in=15.0, alpha=.25, action_size=self.action_size)
        model.compile(loss=lossf, metrics=[pi_lossf, v_lossf], optimizer=opt)
        #         model.summary()

        return model

    def _predict_to_q(self, preds, is_ipiv=True):
        """
        Converts the prediction of the PI-DQN model to Q-values.
        :param preds: original prediction
        :param is_ipiv: boolean indicating whether the prediction is from the IPIV-DQN model
        :return: converted Q-values
        """
        if not is_ipiv:
            return preds
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

    def _gen_noise_only(self, sigma=0.2):
        """
        Generates noise only.
        :param sigma: magnitude of the noise
        :return: vector of noise
        """
        if self.atari:
            size = self.input_shape
        else:
            size = self.state_space
        zeros = np.zeros(size)
        return np.random.uniform(zeros - sigma, zeros + sigma)

    def _gen_uniform_noise(self, input, sigma=0.2):
        """
        Generates a noise vector with uniform distribution.
        :param input: original input
        :param sigma: magnitude of the noise
        :return: noised input vector
        """
        noise = tf.reshape(np.random.uniform(input - sigma, input + sigma), [1, self.state_space])
        return noise

    def get_intervals_size(self, predict):
        """
        Calculates the interval sizes for the given prediction.
        :param predict: prediction
        :return: an array of interval sizes
        """
        return np.array([predict[0][x] - predict[0][x + self.action_size] for x in range(self.action_size)])

    def _is_false_interval(self, predict, int_size=50):
        """
            The function checks if the intervals are smaller than the int_size.
            :param predict: the prediction of the model
            :param int_size: threshold for the intervals size
            :return: boolean indicating whether the intervals are smaller than the int_size
        """
        bounds_len = self.get_intervals_size(predict)
        print(bounds_len)
        return np.any(abs(bounds_len) > int_size)

    def calc_past_avg(self, pred_hist, past_avg_size=20):
        """
            The function calculates average of window in size past_avg_size.
            :param pred_hist: the prediction history
            :param past_avg_size: the size of the window
            :return: the average of the window
        """
        history = np.concatenate(pred_hist, axis=0)[-past_avg_size:]

        avg = history.mean(axis=0)
        avg[-self.action_size:] = np.repeat(0.5, self.action_size)
        return np.reshape(avg, (1, len(pred_hist[-1][0])))

    def calc_w_sum(self, pred_hist, past_avg_size, predict):
        """
            The function calculates the weighted sum of past predictions' average and current prediction
            :param pred_hist: the predictions' history
            :param past_avg_size: the size of the window
            :param predict: the current prediction
            :return: the weighted sum of past average and current prediction
        """
        avg = self.calc_past_avg(pred_hist, past_avg_size)
        # dynamic weight based on the difference between the current prediction and the past average
        ws_weight = 0.75 if np.max(abs(avg - predict)) > 100 else 0.5
        wsum = (ws_weight * avg) + ((1 - ws_weight) * predict)
        #         print(wsum)
        wsum[0][-self.action_size:] = np.repeat(0.5, self.action_size)
        #         print(avg,predict,wsum)
        return wsum

    def update_log_noise(self, predict, update=True):
        """
        Add to the log information about the new interval that is based on average and prediction.
        :param predict: the prediction
        :param update: boolean indicating whether to update the log
        """
        print(len(predict[0]))
        pivenQ1U, pivenQ2U, pivenQ1L, pivenQ2L, pivenQ1v, pivenQ2v,_,_,_,_,_,_= predict[0]
        self.log['Piven-Q1U_Noise'].append(pivenQ1U if update else '-')
        self.log['Piven-Q1L_Noise'].append(pivenQ1L if update else '-')
        self.log['Piven-Q2U_Noise'].append(pivenQ2U if update else '-')
        self.log['Piven-Q2L_Noise'].append(pivenQ2L if update else '-')

    def _need_backup(self, predict, int_size, past_interval):
        """
            The function looks at history predictions and decides what to do if to use the vanilla-DQN model's prediction or not.
            :param predict: the prediction of the model
            :param int_size: the threshold for the intervals size
            :param past_interval: boolean array indicating whether the intervals in the past are smaller than the int_size
        """
        if not self._is_false_interval(predict, int_size):
            return False
        elif len(past_interval) < 3:
            return False
        else:
            return reduce(lambda a, b: a and b, past_interval[-3:])

    def _choose_action(self, predict, is_ipiv=True, pred_hist=[], int_thr=50, past_avg_window=20, backup_predict=None,
                       past_intervals=None, use_past_avg=True):
        """

        The function chooses the best action based in the given policy. The params are seperated to some sections:

        == Base params ==
        :param predict - The prediction that the action choice is based on.
        :param is_ipiv - boolean that indicates if the action is from the ipiv model or not.

        == Cold start evaluation params ==

        :param pred_hist - Array of history predictions.
        :param int_thr - Threshold for interval size.
        :param past_avg_window - Past steps window size for the average calculations.

        == Use noPiven as backup params ==
        :param backup_predict - vanilla-DQN model's prediction for case of using backup
        :param use_past_avg - Boolean that indicates whether to use the past average or not.
        :param past_intervals - Boolean array that indicates whether the intervals in the past are smaller than the int_size

        """

        if is_ipiv:
            if backup_predict is not None:
                if self._need_backup(predict, int_thr, past_intervals):
                    self.update_log_noise(predict, False)
                    return np.argmax(backup_predict)
            if use_past_avg:
                if self._is_false_interval(predict, int_thr):
                    predict = self.calc_w_sum(pred_hist, past_avg_window, predict)
                    self.update_log_noise(predict, True)
                else:
                    self.update_log_noise(predict, False)
            else:
                self.update_log_noise(predict, False)

        q_vals = self._predict_to_q(predict, is_ipiv)

        return np.argmax(q_vals)

    def reset_log(self):
        """
        Initialize the log dictionary, the log dictionary is used to store the noise values and prediction for each action.
        :return: none, but the log dictionary is initialized.
        """
        self.log = {}
        self.log['noPiven-Q1'] = []
        self.log['noPiven-Q2'] = []
        self.log['Piven-Q1U'] = []
        self.log['Piven-Q1L'] = []
        self.log['Piven-Q1v'] = []
        self.log['Piven-Q1U_Noise'] = []
        self.log['Piven-Q1L_Noise'] = []
        self.log['Piven-Q1int'] = []
        self.log['Piven-Q2U'] = []
        self.log['Piven-Q2L'] = []
        self.log['Piven-Q2v'] = []
        self.log['Piven-Q2U_Noise'] = []
        self.log['Piven-Q2L_Noise'] = []
        self.log['Piven-Q2int'] = []
        self.log['noPiven-a'] = []
        self.log['Piven-a'] = []
        self.log['isDiff'] = []

    def test_agent_with_noise_after_cold(self, seed=10, iters=1, noise=0.0, verbose=True, cold_start=20, int_thr=50,
                                         past_avg_window=20, bu_predict=False,
                                         use_past_avg=False):
        """
        Main evaluation function. the function runs both the vanilla-DQN and the ipiv-DQN models and writes the
        results to the log dictionary.
        :param seed: The seed for the random number generator.
        :param iters: The number of iterations to run the evaluation.
        :param noise: The noise value to be used in the evaluation.
        :param verbose: Boolean that indicates whether to print the results or not.
        :param cold_start: The number of steps to run the evaluation before starting the evaluation.
        :param int_thr: The threshold for the interval size.
        :param past_avg_window: The window size for the past average calculations.
        :param bu_predict: Boolean that indicates the usage of "use vanilla-DQN as backup" policy.
        :param use_past_avg: Boolean that indicates the usage of "use past average" policy.


        """

        self.reset_log()
        self.env1.seed(seed)
        self.env2.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        if verbose:
            print(f'========TESTING FOR MODEL {self.name1} VS {self.name2}  , noise:{noise}')

        rewards = []
        for n in range(iters):
            state1 = self.env1.reset()
            self.env2.set_state(self.env1.env)
            if self.atari:
                state1 = process_observation(state1)
                state1 = np.stack([state1] * 4)
                state1 = process_state_batch(state1.reshape((1,) + state1.shape))
                print(state1.shape)
            else:
                state1 = np.reshape(state1, [1, self.state_space])
                
            state2 = state1
            done1 = False
            print1 = True
            done2 = False
            print2 = True
            i = 0
            r1 = 0
            r2 = 0
            predicts = []
            is_false = []
            while not (done1 and done2):
                noise_vec = self._gen_noise_only(sigma=noise)
                if not done1:
                    if not self.model1SA: 
                        predict1 = self.model1.predict(state1)
                        action1 = self._choose_action(predict=predict1, is_ipiv=False)
                    else:
                        state_tensor = torch.from_numpy(np.ascontiguousarray(state1[0])).unsqueeze(0).cuda().to(torch.float32)
                        predict1 = self.model1.forward(state_tensor, method_opt='forward')
                        action1 = self.model1.act(state_tensor)[0]
                    n_state1, reward1, done1, _ = self.env1.step(action1)
                    r1 += reward1


                    if self.atari:
                        n_state1 = process_observation(n_state1)
                        state1 = np.stack([n_state1, state1[0][0], state1[0][1], state1[0][2]])
                        state1 = process_state_batch(state1.reshape((1,) + state1.shape))
                        if i > cold_start:
                            state1 = state1 + noise_vec
                    else:
                        if i > cold_start:
                            n_state1 = n_state1 + noise_vec
                        state1 = np.reshape(n_state1, [1, self.state_space])
                        
                if not done2:
                    predict2 = self.model2.predict(state2)

                
                    if bu_predict:
                        backup_predict = self.bu_model.predict(state2)
                    else:
                        backup_predict = None  

                    action2 = self._choose_action(predict=predict2, pred_hist=predicts, int_thr=int_thr,
                                                  past_avg_window=past_avg_window, backup_predict=backup_predict,
                                                  past_intervals=is_false, use_past_avg=use_past_avg)

                    is_false.append(self._is_false_interval(predict2, int_thr))
                    to_append = predict2 if not is_false[-1] else self.calc_past_avg(predicts, past_avg_window)

                    predicts.append(to_append)

                    n_state2, reward2, done2, _ = self.env2.step(action2)
                    r2 += reward2

                        
                    if self.atari:
                        n_state2 = process_observation(n_state2)
                        state2 = np.stack([n_state2, state2[0][0], state2[0][1], state2[0][2]])
                        state2 = process_state_batch(state2.reshape((1,) + state2.shape))
                        if i > cold_start:
                            state2 = state2 + noise_vec
                    else:
                        if i > cold_start:
                            n_state2 = n_state2 + noise_vec
                        state2 = np.reshape(n_state2, [1, self.state_space])

                if done1 and print1:
                    if verbose:
                     print(f'{self.name1} Done in {r1}')
                    print1 = False
                if done2 and print2:
                    if verbose:
                        print(f'{self.name2} Done in {r2}')
                    print2 = False
                if done2 and not done1:
                    self.update_log_noise(predict2, False)
                self.update_log(predict1, predict2, action1, action2, done1, done2)
                i += 1

            rewards.append(r2)
        if verbose:
            print(f'r1={r1}, r2={r2}')
        return r1, r2

    def update_log(self, predict1, predict2, action1, action2, done1, done2):
        """
        Update the log of the episode
        :param predict1: prediction of agent 1
        :param predict2: prediction of agent 2
        :param action1: action of agent 1
        :param action2: action of agent 2
        :param done1: status of agent 1
        :param done2: status of agent 2
        :return: nothing, updates the log.
        """
        noPivenQ1, noPivenQ2 = predict1[0]
        pivenQ1U, pivenQ2U, pivenQ1L, pivenQ2L, pivenQ1v, pivenQ2v = predict2[0]
        self.log['noPiven-Q1'].append(noPivenQ1 if not done1 else '-')
        self.log['noPiven-Q2'].append(noPivenQ2 if not done1 else '-')
        self.log['Piven-Q1U'].append(pivenQ1U if not done2 else '-')
        self.log['Piven-Q1L'].append(pivenQ1L if not done2 else '-')
        self.log['Piven-Q1v'].append(pivenQ1v if not done2 else '-')
        self.log['Piven-Q1int'].append(pivenQ1U - pivenQ1L if not done2 else '-')
        self.log['Piven-Q2U'].append(pivenQ2U if not done2 else '-')
        self.log['Piven-Q2L'].append(pivenQ2L if not done2 else '-')
        self.log['Piven-Q2v'].append(pivenQ2v if not done2 else '-')
        self.log['Piven-Q2int'].append(pivenQ2U - pivenQ2L if not done2 else '-')
        self.log['noPiven-a'].append(action1 if not done1 else '-')
        self.log['Piven-a'].append(action2 if not done2 else '-')
        self.log['isDiff'].append(not action1 == action2 if not (done1 or done2) else '-')


def eval_agents(vanilla_path, ipiv_path,bu_model_path, count_steps=False, results_path="",verbose=True, cold_start=0, int_thr=0.5,
                past_avg_window=10, vanilla_backup=False, use_past_avg=True,model1SA=False):
    eval_seeds = [5, 7, 10, 18, 21, 26, 30, 33, 39, 45, 47, 49]

    noises = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    results = pd.DataFrame()

    # for noise in tqdm.tqdm(noises):
    for noise in tq.tqdm(noises):

        params_dict = dict(noise=noise, cold_start=cold_start, int_thr=int_thr, past_avg_window=past_avg_window,
                           bu_predict=vanilla_backup, use_past_avg=use_past_avg)
        r1s = []
        r2s = []
        # for seed in tqdm.tqdm(eval_seeds):
        for seed in tq.tqdm(eval_seeds):
            agents_eval = CartpoleIPIVAgentsEval(model1_path=vanilla_path, model2_path=ipiv_path,bu_model_path=bu_model_path
                                                 , name1='vanilla-dqn-agent', name2='ipiv-dqn-agent',model1SA=model1SA)
            r1, r2 = agents_eval.test_agent_with_noise_after_cold(seed=seed, verbose=verbose,
                                                                  **params_dict)
            r1s += [r1]
            r2s += [r2]
            if count_steps:
                print({k: len(v) for k, v in agents_eval.log.items()})
                pd.DataFrame.from_dict(agents_eval.log).to_csv(
                    f'{results_path}/steps_print.csv')
        params_dict['vanilla-DQN-score'] = np.mean(r1s)
        params_dict['IPIV-DQN-score'] = np.mean(r2s)
        params_dict = {key: [params_dict[key]] for key in params_dict.keys()}
        res_df = pd.DataFrame(params_dict)
        results = pd.concat([results, res_df], ignore_index=True)
    results.to_csv(
        f'{results_path}/Evaluation_avgs.csv')
#main function
if __name__ == '__main__':
    p1 = "ataribase/breakout2/dqn_BreakoutDeterministic-v4_weights_4250000.h5"
    p2 = "save_model/Atari/snowy-pine-24/Atari_ddqn_snowy-pine-24_3043_max_score9.0.h5"
    agents_eval = CartpoleIPIVAgentsEval(model1_path=p1, model2_path=p2,bu_model_path=p1
                                        , name1='vanilla-dqn-agent', name2='ipiv-dqn-agent',model1SA=False,atari=True)
    r1, r2 = agents_eval.test_agent_with_noise_after_cold()
    print("Hello")