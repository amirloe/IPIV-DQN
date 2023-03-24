import numpy as np
import pandas as pd
import gym
from onlyRegression import dqn_model,build_piven,piven_loss_no_sign,predict_to_q,_gen_uniform_noise
from cartpole import CartPole



def test_agent_noise_cmp_models(model1,model2,env1,env2,sigma):
    done1 = False
    done2 = False
    score1 = 0
    score2 = 0
    state = env1.reset()
    env2.set_state(env1.env)
    state = np.reshape(state, [1,4])
    state1 = _gen_uniform_noise(state,sigma)
    state2 = state1

    while not (done1 and done2):

        # get action for the current state and go one step in environment
        q_value1 = model1.predict(state1)
        q_value2 = model2.predict(state2)
        q_value2 =  predict_to_q(q_value2)
        action1 = np.argmax(q_value1[0])
        action2 = np.argmax(q_value2[0])
        
        if not done1:
            next_state1, reward1, done1, info1 = env1.step(action1)
            next_state1 = np.reshape(next_state1, [1, 4])
            next_state1 = _gen_uniform_noise(next_state1,sigma)
            score1 += reward1
            state1 = next_state1
        if not done2:
            next_state2, reward2, done2, info2 = env2.step(action2)
            next_state2 = np.reshape(next_state2, [1, 4])
            next_state2 = _gen_uniform_noise(next_state2,sigma)
            score2 += reward2
            state2 = next_state2
        
        
    print(f"Model 1 Done in {score1} episodes. Model 2 Done in {score2} episodes.")
    return score1,score2

def test_agent_noise(train_model,env,sigma,isPiven):
    done = False
    score = 0
    state = env.reset()
    state = np.reshape(state, [1,4])
    state = _gen_uniform_noise(state,sigma)

    while not done:

        # get action for the current state and go one step in environment
        q_value = train_model.predict(state)
        # 
        if isPiven:
            # print(q_value)
            q_value =  predict_to_q(q_value)
        action = np.argmax(q_value[0])
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        next_state = _gen_uniform_noise(next_state,sigma)
        score += reward
        state = next_state
    print(f"Done in {score} episodes.")
    return score

env1 = CartPole()
env2 = CartPole()
model1=dqn_model()
model1.load_weights('save_model/NoPiven_model/cartpole_ddqn_NoPiven_100.h5')
bias_init=np.append(np.repeat(60.,2) ,np.repeat(40.,2))

res = {}
res['att'] = []
res['sigma'] =[]
res['pivenScore'] =[]
res['noPivenScore']= []
# for att in range(3,11,1):
#     model2 = build_piven(bias_init)
#     model2.load_weights(f'only_pi_score_500.0_att{att}.h5')
#     for sigma in np.arange(0.,0.6,0.1):
#         res['att'].append(att)
#         print(f"Sigma == {sigma}")
#         print("NO PIVEN:")
#         s1 = test_agent_noise(model1,env,sigma,False)
#         res['sigma'].append(sigma)
#         res['noPivenScore'].append(s1)
#         print("PIVEN:")
#         s2 = test_agent_noise(model2,env,sigma,True)
#         res['pivenScore'].append(s2)
#         print()
for att in range(3,11,1):
    model2 = build_piven(bias_init)
    model2.load_weights(f'models_reg_only/only_pi_score_500.0_att{att}.h5')
    for sigma in np.arange(0.,0.6,0.1):
        res['att'].append(att)
        print(f"Sigma == {sigma}")
        s1,s2 = test_agent_noise_cmp_models(model1,model2,env1,env2,sigma)
        res['sigma'].append(sigma)
        res['noPivenScore'].append(s1)
        res['pivenScore'].append(s2)
        print(f'NoPiven:{s1} Piven:{s2}')
        print()
df = pd.DataFrame(res)
df.to_csv("res/SecondAttempt.csv")

