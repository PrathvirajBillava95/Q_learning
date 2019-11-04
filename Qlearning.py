import sys
import random
import numpy as np

epsilon = 0.1  # greedy value
alpha = 0.1  # learning factor
gamma = 0.2  # discounting factor
start_state = 1  # start state - 1 , the actual start state is 2
state_size = 16  # no. of states
action_size = 4  # no. of actions
qTable = np.zeros((state_size, action_size), dtype=float)  # stores the q values of each state with respect to actions
rewards = np.zeros((state_size, action_size))  # stores the rewards of each state with respect to actions
actions = np.full((state_size, action_size), -1)  # stores information of the target state for a particular action
np.set_printoptions(suppress=True)

'''
prints the action associated with index a
'''


def print_policy(s, a):
    action_enum = ["West", "North", "East", "South"]
    if actions[s][a] != -1:
        print(s + 1, action_enum[a])


'''
prints the q value of state s and action a : q(s,a)
'''


def print_q_value(s, a):
    action_enum = ["West", "North", "East", "South"]
    if actions[s][a] != -1:
        print(s + 1, action_enum[a], qTable[s][a])


'''
prints the optimal policies of all states
'''


def print_all_optimal_policies(g1, g2, w, f):
    others = [g1, g2, w, f]
    a = -1
    for s in range(state_size):
        if s not in others:
            tmp = actions[s]
            max_q = -1000000
            for i in range(action_size):
                if tmp[i] != -1:
                    # for all the valid actions, calculate the maximum Q value
                    if qTable[s][i] > max_q:
                        max_q = qTable[s][i]
                        a = i
            print_policy(s, a)


'''
prints the optimal path from the start state to goal state
'''


def print_optimal_path(goal1, goal2):
    s = start_state
    prev_state = -1
    a = -1
    while s != goal1 and s != goal2:
        tmp = actions[s]
        max_q = -1000000
        for i in range(action_size):
            if tmp[i] != -1 and tmp[i] != prev_state:
                # for all the valid actions, calculate the maximum Q value
                if qTable[s][i] > max_q:
                    max_q = qTable[s][i]
                    a = i
        print_policy(s, a)
        prev_state = s
        s = actions[s][a]


'''
Initializes the rewards table
'''


def initialize_reward_table(goal1, goal2, forbid):
    for s in range(state_size):
        if s == goal1 or s == goal2:
            rewards[s][0] = 100
        elif s == forbid:
            rewards[s][0] = -100
        else:
            for a in range(action_size):
                if actions[s][a] != -1:
                    rewards[s][a] = -0.1


'''
Initializes the action table
'''


def initialize_action_table(goal1, goal2, wall, forbid):
    for i in range(state_size):
        for j in range(action_size):
            if i == goal1 or i == goal2 or i == forbid:
                actions[i][j] = 16  # consider the value 16 as exit state
            elif i != wall:
                if j == 0:  # move west
                    if i % 4 != 0:
                        actions[i][j] = (i - 1)
                if j == 1:  # move north
                    if i < 12:
                        actions[i][j] = i + 4
                if j == 2:  # move east
                    if i % 4 != 3:
                        actions[i][j] = i + 1
                if j == 3:  # move south
                    if i > 3:
                        actions[i][j] = i - 4


'''
Calculate the q value of state s when action a is taken
Q(s,a) = (1- alpha) Q(s,a) + alpha (R(s, a, s') + gamma (max(Q(s',a')))
alpha: learning rate
gamma: discounting factor
'''


def calculate_q_value(s, a):
    old_q_value = qTable[s][a]
    s_prime = actions[s][a]
    if s_prime == 16:  # if target state is exit, then the Q value = Reward achieved
        return rewards[s][a]
    max_q_value = np.amax(qTable[s_prime])  # max(Q(s',a'))
    return ((1 - alpha) * old_q_value) + alpha * (rewards[s][a] + gamma * max_q_value)


'''
Implements Q learning algorithm
'''


def run_q_learning(goal1, goal2, wall, forbid):
    global epsilon
    tmp = [goal1, goal2, forbid]
    for i in tmp:
        # calculate Q values of exit states
        qTable[i][0] = calculate_q_value(i, 0)

    count = 0
    cng_count = 0

    while count < 10000:
        # Maximum 10000 rounds
        if count % 100 == 0:
            old_q_table = np.copy(qTable)  # Every 100 rounds copy Q table to check for convergence

        next_state = start_state
        prev_state = -1
        while next_state != goal1 and next_state != goal2 and next_state != forbid:
            #  while  exit state is not reached, calculate the Q values of each state either
            #  by exploring the sates or exploiting
            s = next_state

            if random.uniform(0, 1) > epsilon:
                #  Exploit with probability of (1 - epsilon)
                #  i.e. select the action with maximum Q value
                tmp = actions[s]
                max_q = -1000000
                for i in range(action_size):
                    if tmp[i] != -1 and tmp[i] != prev_state:
                        if qTable[s][i] > max_q:
                            max_q = qTable[s][i]
                            a = i
            else:
                #  Explore with probability of epsilon
                #  i.e. randomly select the action
                tmp = actions[s]
                action_list = []
                for i in range(action_size):
                    if tmp[i] != -1 and tmp[i] != prev_state:
                        action_list.append(i)
                a = random.choice(action_list)

            s_prime = actions[s][a]
            qTable[s][a] = calculate_q_value(s, a)  # calculate the Q value for selected action a

            if s_prime != wall:
                #  if the target state is wall, then remain in the current state else move to the target state
                next_state = s_prime

            prev_state = s

        if np.allclose(qTable, old_q_table, atol=0.0):
            # After completing each round, check if the Q values have converged.
            # Compare the old and new Q values, if the values are not changing for a specific period
            # then exit the termination
            cng_count += 1
            if cng_count > 200:
                # print("EXIT:", count)
                break
        else:
            cng_count = 0
        count = count + 1


'''
main function
'''


def main():
    p_flag = False
    q_flag = False
    state = 0

    goal1 = int(sys.argv[1]) - 1  # Goal 1
    goal2 = int(sys.argv[2]) - 1  # Goal 2
    forbid = int(sys.argv[3]) - 1  # Forbidden sqaure
    wall = int(sys.argv[4]) - 1  # Wall

    if sys.argv[5] == "p":
        p_flag = True
    elif sys.argv[5] == "q":
        state = int(sys.argv[6]) - 1
        q_flag = True
    else:
        print("Invalid input")
        return

    initialize_action_table(goal1, goal2, wall, forbid)
    initialize_reward_table(goal1, goal2, forbid)
    run_q_learning(goal1, goal2, wall, forbid)

    if p_flag:
        print_all_optimal_policies(goal1, goal2, wall, forbid)
        # Additional output
        print("\n Path from start state to nearest goal:")
        print_optimal_path(goal1, goal2)
    elif q_flag:
        for a in range(action_size):
            print_q_value(state, a)


if __name__ == "__main__":
    main()
