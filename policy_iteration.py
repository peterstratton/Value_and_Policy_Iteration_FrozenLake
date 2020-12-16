import gym
import numpy as np

RUNS = 1000

def policy_iteration(env):
    """
    Function that implements the policy iteration algorithm. It initializes an
    arbitrary policy, updates the values of the environment states using the
    policy, then uses the values to improve the policy.

    Parameters
    ----------
    arg1 : Open AI gym environment (<class 'gym.wrappers.time_limit.TimeLimit'>)
        Gym environment that allows for deterministic policy

    Returns
    -------
    numpy.array
        The optimal policy for the environment
    """
    num_states = env.nS
    num_actions = env.nA
    state_transition = env.P

    # init state values and policy
    values = np.zeros(num_states)
    policy = np.zeros(num_states)

    # loop until policy converges
    stable = False
    gamma = 0.9
    while not stable:
        values = policy_evaluation(policy, values, state_transition, \
                                   num_states, num_actions, gamma)
        policy, stable = policy_improvement(policy, values, state_transition, \
                                            num_states, num_actions, gamma)

    return policy


def policy_evaluation(policy, values, state_transition, num_states, num_actions, gamma):
    """
    Function that evaluates a policy by determining the values of the
    environment states using the policy.

    Parameters
    ----------
    arg1 : numpy.array
        Policy that dictates the action to be made from a given state
    arg2 : numpy.array
        Values of the states under the given policy
    arg3 : dict
        Dictionary of lists, where
        state_transition[state][action] == [(probability, nextstate, reward,
                                             done), ...]
    arg4 : int
        Number of states in the environment
    arg5 : float
        Discount variable

    Returns
    -------
    numpy.array
        New values of each state
    """
    # threshold that defines when values have converged
    thres = 0.001

    # loop till convergence
    delta = 1
    while delta > thres:
        delta = 0

        # iterate through each state
        for state in range(num_states):
            prev_v = values[state]
            action = policy[state]

            # sum weighted values from next states
            value = 0
            for p_ns, n_state, reward, _ in state_transition[state][action]:
                value += p_ns * (reward + gamma * values[n_state])

            values[state] = value
            delta = max(delta, np.abs(values[state] - prev_v))

    return values


def policy_improvement(policy, values, state_transition, num_states, num_actions, gamma):
    """
    Function that improves the given policy by altering its action from a state
    to go to the next state with the largest value.

    Parameters
    ----------
    arg1 : numpy.array
        Policy that dictates the action to be made from a given state
    arg2 : numpy.array
        Values of the states under the given policy
    arg3 : dict
        Dictionary of lists, where
        state_transition[state][action] == [(probability, nextstate, reward,
                                             done), ...]
    arg4 : int
        Number of states in the environment
    arg5 : float
        Discount variable

    Returns
    -------
    numpy.array
        New policy that now gives actions that maximize the expected reward,
        given the current values
    boolean
        If the policy was changed or not
    """
    stable = True

    # iterate through states to improve the policy
    for state in range(num_states):
        prev_a = policy[state]

        # determine actions that give the highest value
        actions = []
        max = 0
        for action in range(num_actions):
            value = 0
            for p_ns, n_state, reward, _ in state_transition[state][action]:
                value += p_ns * (reward + gamma * values[n_state])

            # update best action
            if max < value:
                policy[state] = action
                max = value

        if policy[state] != prev_a:
            stable = False

    return policy, stable


if __name__ == "__main__":
    env = gym.make('FrozenLake8x8-v0')
    observation = env.reset()

    # obtain optimal policy
    policy = policy_iteration(env)
    print("Optimal policy: " + str(policy))

    # test policy
    win = 0
    lose = 0
    scores = []
    for i in range(RUNS):
        observation = env.reset()
        done = False
        t = 0
        reward_total = 0
        while not done:
            # make actions in the environment
            action = policy[observation]
            observation, reward, done, info = env.step(action)
            reward_total += reward

        # sum wins and losses
        scores.append(reward_total)
        if reward == 1:
            win += 1
        else:
            lose += 1

    # display results
    print("-------------------------------------------------------------------")
    print("Policy resulted in winning: " + str((win / float(RUNS)) * 100) + " percent of the time")
    print("Policy resulted in losing: " + str((lose / float(RUNS)) * 100) + " percent of the time")
    print("-------------------------------------------------------------------")
    print("Mean score = %0.2f" %(np.mean(np.array(scores))))
    print("-------------------------------------------------------------------")
    env.close()
