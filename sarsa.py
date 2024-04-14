import gym, cv2
import numpy as np
import pickle as pkl
# Creating the Environment
cliffEnv = gym.make("CliffWalking-v0")
num_episodes=500
q_table=np.zeros((48,4))
total_reward=0
total_steps=0
def policy(state,epsilon=0.0):
    action =int(np.argmax(q_table[state]))
    if np.random.rand()<epsilon:
        action=int(np.random.randint(low=0,high=4  ,size=1 ))       
    return action
for i in range(num_episodes):
    total_reward=0
    total_steps=0
    done=False
    state=cliffEnv.reset()[0]
    action=policy(state,0.0)
    gamma=0.9
    while not done:
        next_state,reward,done,_,info=cliffEnv.step(action)
        # print(next_state)
        next_action=policy(next_state,0.0)
        q_table[state,action]+=0.1*(reward+ gamma* (q_table[next_state,next_action]-q_table[state,action] ))
        state=next_state
        action=next_action
        total_reward+=reward
        total_steps+=1
    print("Episode: ",i, "Total reward: ",total_reward,"Total steps: ",total_steps)
cliffEnv.close()

pkl.dump(q_table,open("q_table.pkl","wb"))
print("Q-table saved")