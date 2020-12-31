# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import time


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    improve_iteration = 0
    evalue_iteration = 0
    policy_stable = False

    for i in range(max_iterations):
        value_func, e_iter = evaluate_policy(env, gamma, policy, value_func, max_iterations, tol)
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        improve_iteration += 1
        evalue_iteration += e_iter
        if policy_stable:
            break
    return policy, value_func, improve_iteration, evalue_iteration


def evaluate_policy(env, gamma, policy, value_func, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    value_func: np.array
      The value function array
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    iterations = 0
    V = value_func

    for i in range(max_iterations):
        V_old = V.copy()
        delta = 0
        # Note: Loop all state.
        for s in range(env.nS):
            # Note: Given the current policy, update value function.
            a = policy[s]
            V[s] = calculate_expectation(env, s, a, gamma, V_old)
            delta = max(delta, abs(V[s] - V_old[s]))

        # Note: Check whether to exit.
        iterations += 1
        if delta < tol:
            break

    return V, iterations


def improve_policy(env, gamma, value_function, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_stable = True

    # Note: Loop all state.
    for s in range(env.nS):
        old_action = policy[s]
        max_value = None
        # Note: Loop all action.
        for a in range(env.nA):
            # Note: Calculate expected value.
            expectation = calculate_expectation(env, s, a, gamma, value_function)

            # Note: Note down the action that provides the maximum value, and save the action as policy.
            if max_value is None or max_value < expectation:
                max_value = expectation
                policy[s] = a

        if old_action != policy[s]:
            policy_stable = False

    return policy_stable, policy


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    value functions (numpy array), policy (numpy array), iteration (int)
    """
    V = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype='int')

    iteration_cnt = 0
    for i in range(max_iterations):
        delta = 0
        V_old = V.copy()
        # Note: Loop all state.
        for s in range(env.nS):
            max_value = None
            # Note: Loop all action.
            for a in range(env.nA):
                # Note: Calculate Q(s,a).
                expectation = calculate_expectation(env, s, a, gamma, V_old)

                # Note: Note down the action that provides the maximum Q(s,a), and save the action as policy.
                if max_value is None or max_value < expectation:
                    max_value = expectation
                    policy[s] = a

            # Note: Save the maximum Q(s,a) into the value function: V(s) = max_a Q(s,a).
            V[s] = max_value
            delta = max(delta, abs(V_old[s] - V[s]))

        # Note: Check whether to exit.
        iteration_cnt += 1
        if delta < tol:
            break

    return V, policy, iteration_cnt


def calculate_expectation(env, s, a, gamma, val_func):
    expectation = 0
    for prob, next_state, reward, is_terminal in env.P[s][a]:
        if is_terminal:
            expectation += prob * (reward + gamma * 0)  # Note: The value function of terminal state is 0.
        else:
            expectation += prob * (reward + gamma * val_func[next_state])
    return expectation


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    return str_policy
