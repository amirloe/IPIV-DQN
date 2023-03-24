import sys
import traceback

import gym
import pylab
import os
import numpy as np
from collections import deque
import wandb
from keras import backend as K
from pathlib import Path
from models import DoubleDQNAgent
from modeleval import CartpoleIPIVAgentsEval
import pandas as pd
from processing import process_observation,process_state_batch
import sys
EPISODES = 100000
STEPS_TOTAL = 1000000
STACK_SIZE = 4
# root = 'new_attemptJuly2021/'
root = './'


max_avg_thrs = 200
# env_name = 'cartpole'
# env_name = 'lunarlander'
# env_name = 'Atari'
#read env name as the first argument
env_name = sys.argv[1]
save_path =f'{root}save_model/{env_name}/'
if not os.path.exists(save_path):
    os.makedirs(save_path)



def train_agent(agent,env,state_size,action_size,windows,name,isPiven=True,swap=False):
    pi_losses,v_losses,losses,scores, episodes,iters = [], [],[],[],[],[]
    iter=0
    total_steps = 0
    past_w = agent.model.get_weights()
    # NEW METHOD #
    swap_time=10
    print(agent.run_name)
    ##############
    max_score = 0
    max_past_10 = 0
    for e in range(EPISODES):
        done = False
        score = 0
        if isPiven:
            if agent.env_name == 'Atari':
                state = env.reset()
            else:
                state = np.append(env.reset(),np.zeros(agent.extra_f))


        else:
            state = env.reset()
            
            
        
        #TODO CONV adaptation - insert call for process observation
        if agent.env_name == 'Atari':
            state = process_observation(state)
            state = np.stack([state] * STACK_SIZE)
            # print(state.shape)
        else:
            state = np.reshape(state, [-1, agent.state_size])
        steps = list()
        step_count=0
        steps_p = []
        
        total_q_1_U,total_q_1_L = [],[]
        total_q_2_U,total_q_2_L = [],[]
        total_q_1,total_q_2 = [],[]
        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action,q_vals,full_pred = agent.get_action(state)
            if agent.epsilon > agent.epsilon_min: #and len(agent.memory) >= agent.train_start:
                # agent.epsilon *= agent.epsilon_decay    
                a = -float(agent.epsilon_max - agent.epsilon_min) / float(STEPS_TOTAL)
                b = float(agent.epsilon_max)
                value = max(agent.epsilon_min, a * float(total_steps) + b)
                agent.epsilon = value
                
            next_state, reward, done, info = env.step(action)
            # print(state.shape)
            # batch_state = process_state_batch(state.reshape((1,) + state.shape))
            # print(batch_state.shape)
            steps.append(agent.model(state))
            
            if isPiven:
                if not agent.env_name == 'Atari':
                    extras = agent.calc_extras(steps,windows)
                    next_state = np.append(next_state,extras)
            if agent.env_name == 'Atari':
                next_state = process_observation(next_state)
                next_state = np.stack([next_state, state[0], state[1], state[2]])
            else:
                next_state = np.reshape(next_state, [-1, agent.state_size])
            # if an action make the episode end, then gives penalty of -100
            # reward = reward if not done or score == 499 else -100 #Cartpole
            reward = reward  #Lunar lander
            # reward = np.clip(reward, -1., 1.) # Atari

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # every time step do the training
            loss = agent.train_model()
            
            # NEW METHOD #
            if iter%20 == 0 and iter > 0:
                if swap:
                    agent.swap_trainable()
            ##############
            
            if loss is not None:
                # print(loss.history)
                f_loss = loss.history['loss'][0]
                losses.append(f_loss)
                if len(losses)>2:
                    loss_diff = f_loss-losses[-2]
                    if loss_diff > 100:
                        tmp = agent.model.get_weights()
                        agent.model.set_weights(past_w)
                        # agent.model.save_weights(f"{save_path}{agent.run_name}/{env_name}_ddqn_{agent.run_name}_{e}_before_loss_drop.h5")
                        agent.model.set_weights(tmp)
                        
                if isPiven:
                    pi_loss = loss.history['pi_loss'][0]
                    v_loss = loss.history['v_loss'][0]
                    pi_losses.append(pi_loss)
                    v_losses.append(v_loss)
                iters.append(iter)
                pylab.clf()
                pylab.plot(iters, losses, 'g')
                if isPiven:
                    pylab.plot(iters, pi_losses, 'r')
                    pylab.plot(iters, v_losses, 'b')
                pylab.savefig(f"{root}save_graph/{name}/{env_name}_ddqn_{name}_loss.png")
                iter+=1
                
            if q_vals is not None:
                # print(q_vals)
                total_q_1.append(q_vals[0][0])
                total_q_2.append(q_vals[0][1])
                steps_p.append(step_count)
                pylab.clf()
                pylab.plot(steps_p, total_q_1, 'b')
                pylab.plot(steps_p, total_q_2, 'r')
                if isPiven:
                    total_q_1_U.append(full_pred[0][0])
                    total_q_2_U.append(full_pred[0][1])
                    total_q_1_L.append(full_pred[0][2])
                    total_q_2_L.append(full_pred[0][3])
                    pylab.plot(steps_p, total_q_1_U, 'g')
                    pylab.plot(steps_p, total_q_1_L, 'g')
                    pylab.plot(steps_p, total_q_2_U, 'y')
                    pylab.plot(steps_p, total_q_2_L, 'y')
                pylab.savefig(f"{root}save_graph/{name}/{env_name}_ddqn_{name}_Qvals.png")

            score += reward
            state = next_state
            past_w = agent.model.get_weights()

            if done:
                # every episode update the target model to be same with model
                if e % 200 == 0:
                    agent.update_target_model()
                # agent.update_target_model()

                # every episode, plot the play time
                # score = score if score == 500 else score + 100
                
                scores.append(score)
                episodes.append(e)
                pylab.clf()
                pylab.plot(episodes, scores, 'b')
                pylab.savefig(f"{root}save_graph/{name}/{env_name}_ddqn_{name}.png")
                # print("episode:", e, "  score:", score, "  memory length:",
                #       len(agent.memory), "  epsilon:", agent.epsilon)
                wandb.log({"episode": e, "score": score,"epsilon":agent.epsilon})
                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                past_10_avg = np.mean(scores[-min(10, len(scores)):])
                wandb.log({'past_10_avg':past_10_avg})
                
                if score > max_score:
                    max_score = score
                    wandb.log({"max_score":max_score})
                    agent.model.save_weights(f"{save_path}{agent.run_name}/{env_name}_ddqn_{agent.run_name}_{e}_max_score{max_score}.h5")
                if past_10_avg > max_past_10:
                    max_past_10 = past_10_avg
                    wandb.log({"max_past_10_avg":max_past_10})
                    
                if past_10_avg > max_avg_thrs:
                    agent.model.save_weights(f"{save_path}{agent.run_name}/{env_name}_ddqn_{agent.run_name}_{e}_MAXAVG{past_10_avg}.h5")
                    # grid_search(model=f"{save_path}{agent.run_name}/{env_name}_ddqn_{agent.run_name}_{e}_MAXAVG{past_10_avg}.h5",save_path=f"{save_path}{agent.run_name}/{e}_MAXAVG{past_10_avg}")
                    
                if e%10 == 0:
                    agent.model.save_weights(f"{save_path}{agent.run_name}/episodes/{env_name}_ddqn_{agent.run_name}_{e}.h5")
                    # grid_search(model=f"{save_path}{agent.run_name}/episodes/cartpole_ddqn_{agent.run_name}_{e}.h5",save_path=f"{save_path}{agent.run_name}/episodes/{e}")

            step_count+=1
            total_steps += 1
            
            

        # save the model
#         if e % 50 == 0:
#             agent.model.save_weights(f"{save_path}cartpole_ddqn_{agent.run_name}_{e}.h5")

def test_agent(agent,env):
    done = False
    score = 0
    state = np.append(env.reset(),np.zeros(agent.extra_f))
    state = np.reshape(state, [1, agent.state_size])
    steps = list()
    # memory = deque(maxlen=20000) # classic control
    memory = deque(maxlen=1000000) # Atari
    

    while not done:
        if agent.render:
            env.render()
        # get action for the current state and go one step in environment
        action = agent.test_action(state)
        next_state, reward, done, info = env.step(action)
        steps.append(agent.model(state))
        extras = agent.calc_extras(steps,windows)
        next_state = np.append(next_state,extras)
        next_state = np.reshape(next_state, [1, agent.state_size])
        score += reward
        state = next_state
    print(f"Done in {score} episodes.")


def grid_search(model,path1='../experimants/models/picp/no_piven/Piven-Cartpole_AVG>300_seed=9_noPiven.h5',save_path=""):
    # path2 = 'save_model/sigmoidv/cartpole_ddqn_prime-forest-369_978_MAXAVG383.3.h5'
#     path2 = f'save_model/sigmoidv/{model}.h5'
    p1_seed = path1[-13:-11]
    path2=model
    good_seeds = [5, 7, 10, 18, 21, 26, 30, 33, 39, 45, 47, 49]
    cold_starts = [50]
    int_sizes = [40]
    past_avg_sizes = [30]
#     ws_weights = [.1,.5]
    ws_weights = [.5]
    noises = [0.1,0.2,0.3,0.4,0.5]
    results = pd.DataFrame()
    for cold_start in cold_starts:
        for int_size in int_sizes:
            for past_avg_size in past_avg_sizes:
                for ws_weight in ws_weights:
                    for noise in noises:
                        params_dict = dict(noise=noise,cold_start=cold_start,int_size=int_size,past_avg_size=past_avg_size,fix_policy='w_sum',ws_weight=ws_weight)
                        r1s = []
                        r2s = []
                        for i in good_seeds:
                            x = CartpoleIPIVAgentsEval(model1_path=path1,model2_path=path2,bu_model_path=path1,name1='noPiven',name2=model[:-3])
                            r1,r2 = x.test_agent_with_noise_after_cold(seed = i,policy='point_prediction',v=False,**params_dict)
                            r1s += [r1]
                            r2s += [r2]
                            pd.DataFrame.from_dict(x.log).to_csv((f'{save_path}_Episode_example_noise{noise}.csv'))
                        params_dict['no_piven_score'] = np.mean(r1s)
                        params_dict['piven_socre'] = np.mean(r2s)
                        params_dict = {key:[params_dict[key]] for key in params_dict.keys()}
                        res_df = pd.DataFrame(params_dict)
                        results = pd.concat([results,res_df],ignore_index=True)
    results.to_csv(f'{save_path}_New_Evaluation_avgs.csv')


if __name__ == "__main__":
    # In case of CartPole-v1, you can play until 500 time step
    if env_name == 'LunarLander':
        env = gym.make('LunarLander-v2')
    elif env_name == 'MountainCar':
        env = gym.make('MountainCar-v0')
        
    # env = gym.make('LunarLander-v2')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    # print(state_size)        
    # ===== Atari
    # env = gym.make('BreakoutDeterministic-v4')
    # INPUT_SHAPE = (84, 84)
    # input_shape = (STACK_SIZE,) + INPUT_SHAPE
    # state_size = input_shape
    # ==== END Atari
    action_size = env.action_space.n
    name= "test_sep"
    try:
        os.makedirs(f'{root}save_graph/{name}')
    except:
        pass    
    bias_init=np.append(np.repeat(80.,action_size) ,np.repeat(30.,action_size))
    policy = "point_prediction"
    isPiven=True
    n_windows=0
    windows=[]
    # beta = 0.9
    beta = 0.05 # AMIR!! TEST!! CHANGE FOR OUTHER RESULTS!!!
#     beta = 0
    alpha = 0.15
    gama=0.99
    # lr = 0.0001 #Cartpole
    # lr = 0.0001 # LunarLander
    lr = 0.00025 # LunarLander
    
    
    seed=1255

    ## NEW METHOD ##
    swap = False
    clip = True
    
    agent = DoubleDQNAgent(state_size, action_size,windows=n_windows,isPiven=isPiven,policy=policy,lr=lr,gama=gama,piven_beta=beta,piven_alpha=alpha,bias_init=bias_init,load_model=False,load_name=name,seed=seed,swap=swap,clip = clip,env_name=env_name,pretrained=False)
    Path(f"{save_path}{agent.run_name}/episodes").mkdir(parents=True, exist_ok=True)
    train_agent(agent,env,state_size,action_size,windows=windows,name=name,isPiven=isPiven,swap=swap)
    ################
    for lr in [0.0001,0.001]:
        for gama in [0.99,0.9,0.95]: 
            for alpha in [0.15,0.05,0.25]:
                for beta in [0.9,0.1,0.5]:
    # for bias_init in [np.append(np.repeat(40.,2) ,np.repeat(0.,2)),np.append(np.repeat(20.,2) ,np.repeat(-20.,2)),np.append(np.repeat(10.,2) ,np.repeat(0.,2))]:
    # while True:
                                            try:
                                                print("\n" * 5)
                                                print("=" * 10, "lr", lr, "gama", gama, "alpha", alpha, "beta", beta, "=" * 10)
                                                env.seed(seed)
                                                agent = DoubleDQNAgent(state_size, action_size,windows=n_windows,isPiven=isPiven,policy=policy,lr=lr,gama=gama,piven_beta=beta,piven_alpha=alpha,bias_init=bias_init,load_model=False,load_name=name,seed=seed,swap=swap,clip = clip,env_name=env_name,pretrained=False)
                                                Path(f"{save_path}{agent.run_name}/episodes").mkdir(parents=True, exist_ok=True)
                                                train_agent(agent,env,state_size,action_size,windows=windows,name=name,isPiven=isPiven,swap=swap)
                                                K.clear_session()
                                                agent.run.finish()
                                                seed = seed+1
                                            except Exception as e:
                                                print(("=" * 10) + "Error: " + str(e)+("=" * 10))
                                                traceback.print_exc()
                                                print(e)
                                                print(f"Failed Seed {seed}")
                                                K.clear_session()
                                                agent.run.finish()
                                                seed = seed+1
                                                continue
    # # # for x in range(50,950,50):
    # x = 900
    # print(f'============================================ X = {x} ======================================================')
    # agent = DoubleDQNAgent(state_size, action_size,windows=n_windows,isPiven=True,policy=policy,lr=lr,piven_beta=beta,bias_init=bias_init,load_model=True,load_name=name,load_episode=100)
    # test_agent(agent=agent,env=env)
    # print(f'============================================ ======= ======================================================')

    