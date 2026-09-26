from collections import deque
from itertools import product

import numpy as np

from game_env import GameEnv
from game_state import GameState
"""
solution.py

This file implements value iteration and policy iteration for CrystalRover.

Generative AI declaration: OpenAI Codex (GPT-5, 25 September 2026) was used
to help design, implement, debug, and document the transition model and the
VI/PI algorithms. The author reviewed and tested the resulting code.

You should implement each of the method stubs below. You may add additional methods and/or classes to this file if you 
wish. You may also create additional source files and import to this file if you wish.

COMP3702 Assignment 2 "CrystalRover" Support Code

Last updated by vp 09/09/2026
"""


class Solver:

    STUDENT_NAME = "Yuyan Gong"
    STUDENT_ID = "49079986"
    GITHUB_USERNAME = "yuyan"

    def __init__(self, game_env: GameEnv):
        self.game_env = game_env
        self.states = []
        self.state_set = set()
        self.nonterminal_states = []
        self.valid_actions = {}
        self.transitions = {}
        self.vi_values = {}
        self.vi_policy = {}
        self.vi_delta = float("inf")
        self.pi_values = {}
        self.pi_policy = {}
        self.pi_stable = False
        self.state_index = {}

    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        # TODO: modify below if desired (e.g. disable larger testcases if you're having problems with RAM usage, etc)
        return [1, 2, 3, 4, 5]

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        self._build_model()
        self.vi_values = {state: 0.0 for state in self.states}
        self.vi_policy = {
            state: self.valid_actions[state][0] for state in self.nonterminal_states
        }
        self.vi_delta = float("inf")

    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self.vi_delta <= self.game_env.epsilon

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """
        delta = 0.0
        for state in self.nonterminal_states:
            old_value = self.vi_values[state]
            action, value = self._best_action(state, self.vi_values)
            self.vi_values[state] = value
            self.vi_policy[state] = action
            delta = max(delta, abs(value - old_value))
        self.vi_delta = delta

    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while True:
            self.vi_iteration()

            # NOTE: vi_iteration is always called before vi_is_converged
            if self.vi_is_converged():
                break

    def vi_get_state_value(self, state: GameState):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        return self.vi_values.get(state, 0.0)

    def vi_select_action(self, state: GameState):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ACTIONS)
        """
        if state in self.vi_policy:
            return self.vi_policy[state]
        actions = self._actions_for_state(state)
        return actions[0]

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        self._build_model()
        self.pi_values = {state: 0.0 for state in self.states}
        self.pi_policy = {}
        for state in self.nonterminal_states:
            preferred = (GameEnv.JUMP_RIGHT if self._is_crater(state)
                         else GameEnv.WALK_RIGHT)
            actions = self.valid_actions[state]
            self.pi_policy[state] = preferred if preferred in actions else actions[0]
        self.pi_stable = False

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self.pi_stable

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        # Exact policy evaluation solves (I - gamma P_pi)v = r_pi.
        count = len(self.nonterminal_states)
        matrix = np.eye(count, dtype=float)
        rewards = np.zeros(count, dtype=float)
        for state in self.nonterminal_states:
            row = self.state_index[state]
            action = self.pi_policy[state]
            for probability, next_state, reward in self.transitions[(state, action)]:
                rewards[row] += probability * reward
                if not self._is_terminal(next_state):
                    matrix[row, self.state_index[next_state]] -= (
                        self.game_env.gamma * probability
                    )
        evaluated = np.linalg.solve(matrix, rewards)
        for state, row in self.state_index.items():
            self.pi_values[state] = float(evaluated[row])

        stable = True
        for state in self.nonterminal_states:
            old_action = self.pi_policy[state]
            new_action, _ = self._best_action(state, self.pi_values)
            self.pi_policy[state] = new_action
            if new_action != old_action:
                stable = False
        self.pi_stable = stable

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while True:
            self.pi_iteration()

            # NOTE: pi_iteration is always called before pi_is_converged
            if self.pi_is_converged():
                break

    def pi_select_action(self, state: GameState):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ACTIONS)
        """
        if state in self.pi_policy:
            return self.pi_policy[state]
        return self._actions_for_state(state)[0]

    # === Helper Methods ===============================================================================================
    def _is_terminal(self, state):
        return self.game_env.is_solved(state) or self.game_env.is_game_over(state)

    def _is_crater(self, state):
        return self.game_env.grid_data[state.row][state.col] == GameEnv.CRATER_TILE

    def _actions_for_state(self, state):
        allowed = (GameEnv.JUMP_ACTIONS if self._is_crater(state)
                   else GameEnv.WALK_ACTIONS | GameEnv.BOOST_ACTIONS)
        # Preserve the environment's action order so equal-valued actions are
        # resolved consistently across Python processes.
        return tuple(action for action in GameEnv.ACTIONS if action in allowed)

    def _build_model(self):
        if self.states:
            return

        initial = self.game_env.get_init_state()
        queue = deque([initial])
        self.state_set = {initial}

        while queue:
            state = queue.popleft()
            self.states.append(state)
            if self._is_terminal(state):
                continue
            actions = self._actions_for_state(state)
            self.valid_actions[state] = actions
            for action in actions:
                outcomes = self._transition_outcomes(state, action)
                self.transitions[(state, action)] = outcomes
                for probability, next_state, _ in outcomes:
                    if probability > 0.0 and next_state not in self.state_set:
                        self.state_set.add(next_state)
                        queue.append(next_state)

        self.nonterminal_states = [s for s in self.states if not self._is_terminal(s)]
        self.state_index = {state: index for index, state in enumerate(self.nonterminal_states)}

    def _transition_outcomes(self, state, action):
        """Enumerate P(s', r | s, a), merging identical state/reward outcomes."""
        drift = self.game_env.random_drift_prob
        movement_choices = [(action, 1.0 - drift)]
        perpendiculars = self.game_env.PERPENDICULAR_ACTIONS[action]
        movement_choices.extend((a, drift / 2.0) for a in perpendiculars)

        double = self.game_env.random_double_prob
        sequence_choices = []
        for movement, probability in movement_choices:
            sequence_choices.append(((movement,), probability * (1.0 - double)))
            sequence_choices.append(((movement, movement), probability * double))

        merged = {}
        for sequence, sequence_probability in sequence_choices:
            distance_options = []
            for movement in sequence:
                if movement in GameEnv.BOOST_ACTIONS:
                    distance_options.append(tuple(enumerate(self.game_env.boost_probabilities)))
                else:
                    distance_options.append(((1, 1.0),))

            for sampled in product(*distance_options):
                probability = sequence_probability
                current = state
                total_reward = 0.0
                for movement, (distance, distance_probability) in zip(sequence, sampled):
                    probability *= distance_probability
                    current, reward, game_over = self._apply_dynamics_with_distance(
                        current, movement, distance
                    )
                    total_reward += reward
                    if game_over:
                        break
                key = (current, round(total_reward, 12))
                merged[key] = merged.get(key, 0.0) + probability

        return tuple((prob, next_state, reward)
                     for (next_state, reward), prob in merged.items()
                     if prob > 0.0)

    def _apply_dynamics_with_distance(self, state, action, move_distance):
        # A doubled action can become invalid after its first movement (for
        # example, a walk that enters a crater). The simulator ignores that
        # invalid second movement without charging its action cost.
        if action in GameEnv.JUMP_ACTIONS and not self._is_crater(state):
            return state, 0.0, False
        if action in (GameEnv.WALK_ACTIONS | GameEnv.BOOST_ACTIONS) and self._is_crater(state):
            return state, 0.0, False

        reward = -self.game_env.ACTION_COST[action]
        direction = self.game_env._action_direction(action)
        delta_row, delta_col = {
            'LEFT': (0, -1), 'RIGHT': (0, 1),
            'UP': (-1, 0), 'DOWN': (1, 0),
        }[direction]

        next_row, next_col = state.row, state.col
        for _ in range(move_distance):
            candidate_row = next_row + delta_row
            candidate_col = next_col + delta_col
            if (not (0 <= candidate_row < self.game_env.n_rows and
                     0 <= candidate_col < self.game_env.n_cols) or
                    self.game_env.grid_data[candidate_row][candidate_col] == GameEnv.ROCK_TILE):
                reward -= self.game_env.collision_penalty
                break
            next_row, next_col = candidate_row, candidate_col
            tile = self.game_env.grid_data[next_row][next_col]
            if tile == GameEnv.CRATER_TILE:
                break
            if tile == GameEnv.LAVA_TILE:
                reward -= self.game_env.game_over_penalty
                break

        crystal_status = state.crystal_status
        position = (next_row, next_col)
        if position in self.game_env.crystal_positions:
            index = self.game_env.crystal_positions.index(position)
            if crystal_status[index] == 0:
                updated = list(crystal_status)
                updated[index] = 1
                crystal_status = tuple(updated)

        next_state = GameState(next_row, next_col, crystal_status)
        return next_state, reward, self.game_env.is_game_over(next_state)

    def _q_value(self, state, action, values):
        total = 0.0
        for probability, next_state, reward in self.transitions[(state, action)]:
            continuation = 0.0 if self._is_terminal(next_state) else values.get(next_state, 0.0)
            total += probability * (reward + self.game_env.gamma * continuation)
        return total

    def _best_action(self, state, values):
        actions = self.valid_actions[state]
        best_action = actions[0]
        best_value = self._q_value(state, best_action, values)
        for action in actions[1:]:
            value = self._q_value(state, action, values)
            if value > best_value + 1e-12:
                best_action, best_value = action, value
        return best_action, best_value

