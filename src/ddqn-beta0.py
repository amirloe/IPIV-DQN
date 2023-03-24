import sys
import gym
import pylab
import os
import numpy as np
from collections import deque
import wandb

from models import DoubleDQNAgent
EPISODES = 1000
# root = 'new_attemptJuly2021/'
root = './'



def train_agent(agent,env,state_size,action_size,windows,name,isPiven=True,swap=False):
    pi_losses,v_losses,losses,scores, episodes,iters = [], [],[],[],[],[]
    iter=0
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
            state = np.append(env.reset(),np.zeros(agent.extra_f))
        else:
            state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
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
                agent.epsilon *= agent.epsilon_decay
            next_state, reward, done, info = env.step(action)
            steps.append(agent.model(state))
            
            if isPiven:
                extras = agent.calc_extras(steps,windows)
                next_state = np.append(next_state,extras)

            next_state = np.reshape(next_state, [1, agent.state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

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
                pylab.savefig(f"{root}save_graph/{name}/cartpole_ddqn_{name}_loss.png")
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
                pylab.savefig(f"{root}save_graph/{name}/cartpole_ddqn_{name}_Qvals.png")

            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.clf()
                pylab.plot(episodes, scores, 'b')
                pylab.savefig(f"{root}save_graph/{name}/cartpole_ddqn_{name}.png")
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
                    agent.model.save_weights(f"{root}save_model/largev/cartpole_ddqn_{agent.run_name}.h5")
                if past_10_avg > max_past_10:
                    max_past_10 = past_10_avg
                    wandb.log({"max_past_10_avg":max_past_10})
                    
                if past_10_avg > 300:
                    agent.model.save_weights(f"{root}save_model/largev/cartpole_ddqn_{agent.run_name}_{e}_MAXAVG{past_10_avg}.h5")
                    
#                 if past_10_avg > 250:
#                     agent.model.save_weights(f"{root}save_model/good/cartpole_ddqn_{name}_{e}_AVGPASSTEST.h5")
#                     # sys.exit()
                if score == 500: #FOR gilad request
                    agent.model.save_weights(f"{root}save_model/largev/cartpole_ddqn_{agent.run_name}_{e}_REACHMAXSCORE.h5")
                    
                    # sys.exit()
            step_count+=1
            

        # save the model
#         if e % 50 == 0:
#             agent.model.save_weights(f"{root}save_model/cartpole_ddqn_{name}_{e}.h5")

def test_agent(agent,env):
    done = False
    score = 0
    state = np.append(env.reset(),np.zeros(agent.extra_f))
    state = np.reshape(state, [1, agent.state_size])
    steps = list()
    memory = deque(maxlen=20000)

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
    

if __name__ == "__main__":
    # In case of CartPole-v1, you can play until 500 time step
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    print(state_size)
    action_size = env.action_space.n
    name= "test_sep"
    try:
        os.mkdir(f'{root}save_graph/{name}')
    except:
        pass    
    bias_init=np.append(np.repeat(75.,2) ,np.repeat(40.,2))
    policy = "point_prediction"
    isPiven=True
    n_windows=0
    windows=[]
    beta = 0.9
#     beta = 0
    alpha = 0.05
    gama=0.99
    lr = 0.001
    seed=6668
    ## NEW METHOD ##
    swap = False
    clip = True
    largev = True
    ################
    for lr in [0.0001,0.001]:
        for gama in [0.99,0.9,0.95]: 
            for alpha in [0.05,0.15,0.25]:
#                 for beta in [0.9,0.1,0.5]:
#     for bias_init in [np.append(np.repeat(40.,2) ,np.repeat(0.,2)),np.append(np.repeat(20.,2) ,np.repeat(-20.,2)),np.append(np.repeat(10.,2) ,np.repeat(0.,2))]:
                agent = DoubleDQNAgent(state_size, action_size,windows=n_windows,isPiven=isPiven,policy=policy,lr=lr,gama=gama,piven_beta=beta,piven_alpha=alpha,bias_init=bias_init,load_model=False,load_name=name,seed=seed,swap=swap,clip = clip,largev=largev)
                train_agent(agent,env,state_size,action_size,windows=windows,name=name,isPiven=isPiven,swap=swap)
                agent.run.finish()
    # # for x in range(50,950,50):
    # x = 900
    # print(f'============================================ X = {x} ======================================================')
    # agent = DoubleDQNAgent(state_size, action_size,windows=n_windows,isPiven=True,policy=policy,lr=lr,piven_beta=beta,bias_init=bias_init,load_model=True,load_name=name,load_episode=100)
    # test_agent(agent=agent,env=env)
    # print(f'============================================ ======= ======================================================')

    