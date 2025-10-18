#!/usr/bin/env python
"""
CSCI 5302 HW2: Tabular Solution Implementation

This module implements various reinforcement learning algorithms for solving gridworld
problems using tabular methods. It includes value iteration, deterministic policy iteration,
and stochastic policy iteration approaches.

The main components are:
- TabularPolicy: A class representing discrete state-action policies
- GridworldSolver: A class that handles policy computation and visualization
"""

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from hw4_rl.envs import GridworldEnv

student_name = "Yusif Razzaq"  # Set to your name
GRAD = True  # Set to True if graduate student


class TabularPolicy:
    """
    A tabular policy implementation for discrete state/action spaces.

    This class implements a tabular policy and value function for reinforcement learning
    in discrete state/action spaces. It maintains mappings from states to values and
    from states to action probability distributions.

    Attributes:
        num_states (int): Total number of discrete states
        num_actions (int): Total number of possible actions
        state_ranges (np.ndarray): Array of [min, max) ranges for each state dimension
        _value_function (np.ndarray): Array mapping states to their values
        _policy (np.ndarray): Array mapping states to action probability distributions
    """

    def __init__(self, n_states: int, state_ranges: np.ndarray, n_actions: int) -> None:
        """
        Initialize the tabular policy.

        Args:
            n_states: Number of discrete states
            state_ranges: Array of [min, max) ranges for each state dimension
            n_actions: Number of possible actions
        """
        self.num_states = n_states
        self.num_actions = n_actions
        self.state_ranges = state_ranges
        
        # Create data structure to store mapping from state to value
        self._value_function = np.zeros(shape=n_states)

        # Create data structure to store array with probability of each action for each state
        # Initialize with random values and normalize to create proper probability distributions
        self._policy = np.random.uniform(0, 1, size=(n_states, self.num_actions))
        # Normalize each row (state) to sum to 1.0
        self._policy = self._policy / self._policy.sum(axis=1, keepdims=True)

    def get_action(self, state: Union[int, np.ndarray]) -> int:
        """
        Sample an action from the policy's action distribution for the given state.

        Args:
            state: The current state index or state coordinate vector

        Returns:
            The sampled action index
        """
        # Convert state to integer if it's a numpy array
        if isinstance(state, np.ndarray):
            state = self.get_state_index_from_coordinates(state)
        prob_dist = np.array(self._policy[state])
        assert prob_dist.ndim == 1

        # Sample from policy distribution for state
        idx = np.random.multinomial(1, prob_dist / np.sum(prob_dist))
        return np.argmax(idx)

    def set_state_value(self, state: int, value: float) -> None:
        """
        Set the value for a given state.

        Args:
            state: The state index
            value: The value to set for this state
        """
        self._value_function[state] = value

    def get_state_value(self, state: Union[int, np.ndarray]) -> float:
        """
        Get the value for a given state.

        Args:
            state: Either a state index or state coordinate vector

        Returns:
            The value for the given state
        """
        if isinstance(state, int):
            return self._value_function[state]
        else:
            # Map state vector to state index
            return self._value_function[self.get_state_index_from_coordinates(state)]

    def get_state_index_from_coordinates(self, state: np.ndarray) -> int:
        """
        Convert a state coordinate vector to its corresponding state index.

        Args:
            state: A numpy array containing the (x,y) coordinates of the state

        Returns:
            The integer index corresponding to the state coordinates
        """
        # Convert numpy array to tuple of integers
        if isinstance(state, np.ndarray):
            state = tuple(state.astype(int))
        return state[0] * (self.state_ranges[0][1] - self.state_ranges[0][0]) + state[1]

    def get_coordinates_from_state_index(self, state_idx: int) -> np.ndarray:
        """
        Convert a state index to its corresponding coordinate vector.

        Args:
            state_idx: The integer index of the state

        Returns:
            A numpy array containing the (x,y) coordinates corresponding to the state index
        """
        return np.array(
            [
                state_idx // (self.state_ranges[0][1] - self.state_ranges[0][0]),
                state_idx % (self.state_ranges[0][1] - self.state_ranges[0][0]),
            ]
        )

    def get_value_function(self) -> np.ndarray:
        """
        Get a deep copy of the current value function.

        Returns:
            A numpy array representing the value function table, where each entry
            maps a state index to its value
        """
        return copy.deepcopy(self._value_function)

    def set_value_function(self, v: np.ndarray) -> None:
        """
        Set the value function to a new array.

        Args:
            v: A numpy array containing the new value function table
        """
        self._value_function = copy.copy(v)

    def set_policy(self, state: int, action_prob_array: np.ndarray) -> None:
        """
        Set the action probability distribution for a given state.

        Args:
            state: The state index
            action_prob_array: A numpy array containing probabilities for each action
        """
        self._policy[state] = copy.copy(action_prob_array)

    def get_policy(self, state: Union[int, np.ndarray]) -> np.ndarray:
        """
        Get the action probability distribution for a given state.

        Args:
            state: Either a state index or state coordinate vector

        Returns:
            A numpy array containing probabilities for each action in the given state
        """
        if isinstance(state, int):
            return self._policy[state]
        else:
            # Map state vector to state index
            return self._policy[self.get_state_index_from_coordinates(state)]

    def get_policy_function(self) -> np.ndarray:
        """
        Get a deep copy of the current policy function.

        Returns:
            A numpy array representing the policy table, where each entry maps
            a state to a probability distribution over actions
        """
        return copy.deepcopy(self._policy)

    def set_policy_function(self, p: np.ndarray) -> None:
        """
        Set the policy function to a new array.

        Args:
            p: A numpy array containing the new policy table
        """
        self._policy = copy.copy(p)


class GridworldSolver:
    """
    A solver for gridworld reinforcement learning problems.

    This class implements various policy computation methods for gridworld environments,
    including deterministic value iteration, stochastic policy iteration, and
    deterministic policy iteration.

    Attributes:
        _policy_type (str): Type of policy computation method to use
        env (gym.Env): The gridworld environment
        env_name (str): Name of the environment
        temperature (float): Temperature parameter for stochastic policies
        eps (float): Small constant for numerical stability
        gamma (float): Discount factor for future rewards
        solver (TabularPolicy): The policy object that stores computed policies and values
        performance_history (List[float]): History of cumulative rewards from policy evaluations
    """

    def __init__(
        self,
        policy_type: str = "vi",
        gridworld_map_number: int = 0,
        noisy_transitions: bool = False,
    ) -> None:
        """
        Initialize the GridworldSolver.

        Args:
            policy_type: The type of policy computation to use. Must be one of:
                ["vi", "stochastic_pi", "pi"]
            gridworld_map_number: Which gridworld map to use (0 or 1)
            noisy_transitions: Whether to use noisy state transitions
            max_ent_temperature: Temperature parameter for stochastic policies

        Raises:
            AssertionError: If policy_type is not one of the allowed values
        """
        self._policy_type = policy_type
        assert policy_type in ["vi", "stochastic_pi", "pi"]
        self.env: Optional[gym.Env] = None
        self.env_name = ""
        self.init_environment(gridworld_map_number, noisy_transitions)
        self.theta = 1e-3 #parameter to determine when to stop policy evaluation
        self.gamma = 0.99 #future return discount factor
        self.eps = 1e-6

        # Get the unwrapped environment to access its attributes
        assert self.env is not None
        unwrapped_env = self.env.unwrapped
        self.solver = TabularPolicy(
            unwrapped_env.num_states,
            unwrapped_env.get_state_ranges(),
            unwrapped_env.num_actions,
        )
        self.performance_history: List[float] = []

    def init_environment(
        self, gridworld_map_number: int = 0, noisy_transitions: bool = False
    ) -> None:
        """
        Initialize the gridworld environment.

        Args:
            gridworld_map_number: Which gridworld map to use (0 or 1)
            noisy_transitions: Whether to use noisy state transitions

        Raises:
            AssertionError: If gridworld_map_number is not 0 or 1
        """
        assert gridworld_map_number in [0, 1]
        if noisy_transitions:
            self.env_name = f"gridworldnoisy-v{gridworld_map_number}"
        else:
            self.env_name = f"gridworld-v{gridworld_map_number}"

        self.env = gym.make(self.env_name)
        self.env.reset()

    def compute_policy(self) -> None:
        """
        Compute optimal policy using the specified algorithm.
        
        This method selects and runs the appropriate policy computation algorithm based on
        the policy_type specified during initialization.
        """
        if self._policy_type == "vi":
            self._value_iteration()
        else:  # pi
            self._deterministic_policy_iteration()

    def solve(
        self,
        start_state: Optional[np.ndarray] = None,
        visualize: bool = False,
        max_steps: float = float("inf"),
    ) -> Tuple[float, int]:
        """
        Execute the current policy in the environment.
        
        This method runs the current policy from a given start state (or default start state)
        and returns the cumulative reward and number of steps taken.

        Args:
            start_state: Optional starting state coordinates
            visualize: Whether to render the environment
            max_steps: Maximum number of steps to take

        Returns:
            Tuple of (cumulative_reward, num_steps)
        """
        assert self.env is not None
        state, _ = self.env.reset()
        if start_state is not None:
            self.env.unwrapped.change_start_state(start_state)
            state = start_state

        if visualize:
            self.env.render()

        episode_reward = 0
        num_steps = 0
        done = False

        while not done and num_steps < max_steps:
            # Get action using current policy
            action = self.solver.get_action(state)

            # Execute action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            state = next_state
            done = terminated or truncated

            episode_reward += reward
            num_steps += 1

            if visualize:
                self.env.render()

        return episode_reward, num_steps

    def plot_policy_curve(
        self, reward_history: List[float], filename: Optional[str] = None
    ) -> None:
        """
        Plot the learning curve showing policy performance over iterations.

        Args:
            reward_history: List of rewards from each policy evaluation
            filename: Optional path to save the plot
        """
        plt.figure()
        plt.plot(range(len(reward_history)), reward_history)
        plt.xlabel("Iteration")
        plt.ylabel("Return")
        plt.title(f"Policy Iteration Performance - {self.env_name} - gamma={self.gamma}")

        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "..", "saved_figures")
            os.makedirs(figures_dir, exist_ok=True)
            subdir = os.path.join(figures_dir, f"{self.env_name}/{self._policy_type}")
            os.makedirs(subdir, exist_ok=True)
            filename = os.path.join(subdir, f"gamma={self.gamma}_learning.png")

        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_value_function(
        self, value_function: np.ndarray, filename: Optional[str] = None
    ) -> Tuple[np.ndarray, Figure]:
        """
        Plot the value function as a heatmap with proper gridworld visualization.

        Args:
            value_function: Array of state values to plot
            filename: Optional path to save the plot

        Returns:
            Tuple of (image_array, matplotlib_figure)
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        canvas = FigureCanvas(fig)

        # Get the gridworld map from the environment
        unwrapped_env = self.env.unwrapped
        grid_map = unwrapped_env.start_grid_map.copy()
        rows, cols = grid_map.shape
        
        # Create a mask for walls (value 0) and terminal states (3,4) - these will be black
        wall_mask = np.any([grid_map == val for val in [0, 3, 4]], axis=0)
        
        # Use raw value function values without normalization
        V_grid = value_function.reshape(rows, cols)
        
        # Get min/max for colorbar display (excluding walls and terminal states)
        non_wall_non_terminal_values = value_function[~wall_mask.flatten()]
        if len(non_wall_non_terminal_values) > 0:
            v_min = non_wall_non_terminal_values.min()
            v_max = non_wall_non_terminal_values.max()
        else:
            v_min = v_max = 0
        
        # Create the heatmap with raw values
        im = ax.imshow(V_grid, cmap='viridis', vmin=v_min, vmax=v_max, aspect='equal')
        
        # Set black color for walls
        V_grid_masked = V_grid.copy()
        V_grid_masked[wall_mask] = np.nan  # Use NaN for walls so they appear black
        
        # Update the image with masked values
        im.set_array(V_grid_masked)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", size=0)
        
        # Hide ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Find and highlight start state - get it from the environment
        start_state = unwrapped_env.get_start_state()
        if start_state:
            # Outline start state with green border
            rect = plt.Rectangle((start_state[1]-0.5, start_state[0]-0.5), 1, 1, 
                               fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(rect)
        
        # Add reward values for terminal states
        for i in range(rows):
            for j in range(cols):
                if grid_map[i, j] == 0:
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                      facecolor='lightgrey', edgecolor='black', hatch='x')

                    ax.add_patch(rect)
                if start_state and i == start_state[0] and j == start_state[1]:
                    ax.text(j, i, 'S', ha='center', va='center', 
                               fontsize=12, fontweight='bold', color='green')
                elif grid_map[i, j] == 3:  # Positive terminal (green)
                    ax.text(j, i, '+10', ha='center', va='center', 
                           fontsize=12, fontweight='bold', color='green')
                elif grid_map[i, j] == 4:  # Negative terminal (red)
                    ax.text(j, i, '-10', ha='center', va='center', 
                           fontsize=12, fontweight='bold', color='red')
                elif grid_map[i, j] == 5:  # Small negative reward (purple)
                    ax.text(j, i, '-1', ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='purple')
        
        # Set labels and title
        # ax.set_xlabel('X Coordinate')
        # ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Value Function - {self.env_name} - gamma={self.gamma}')
        
        # Add colorbar with actual value range
        # cbar = plt.colorbar(im, ax=ax)
        # cbar.set_label(f'Value (Range: {v_min:.2f} to {v_max:.2f})')
        
        # Invert y-axis so (0,0) is at top-left
        ax.invert_yaxis()
        
        # Save plot
        if filename is None:
            figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../saved_figures")
            os.makedirs(os.path.join(figures_dir, f"{self.env_name}/{self._policy_type}"), exist_ok=True)
            filename = os.path.join(figures_dir, f"{self.env_name}/{self._policy_type}/gamma={self.gamma}_value.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')

        # Convert to image array
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba()).reshape(
            int(fig.get_size_inches()[1] * fig.get_dpi()),
            int(fig.get_size_inches()[0] * fig.get_dpi()),
            4,
        )[:, :, :3]

        return image, fig
    
    def plot_policy(self, policy: Optional[np.ndarray] = None, filename: Optional[str] = None, ax: Optional[plt.Axes] = None) -> None:
        """
        Plot the policy as arrows on the grid.
        
        Args:
            policy: Optional policy table to plot (defaults to solver policy)
            filename: Optional path to save the plot
        """
        if policy is None:
            policy = self.solver.get_policy_function()

        # Get dimensions of the grid
        rows = self.solver.state_ranges[0][1] - self.solver.state_ranges[0][0]
        cols = self.solver.state_ranges[1][1] - self.solver.state_ranges[1][0]

        if ax is None:
            fig, ax = plt.subplots(figsize=(cols, rows))
            ax.set_xlim(0, cols)
            ax.set_ylim(0, rows)
            ax.set_xticks(np.arange(0, cols + 1, 1))
            ax.set_yticks(np.arange(0, rows + 1, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True)
        else:
            fig = ax.get_figure()

        unwrapped_env = self.env.unwrapped

        # Arrow directions (flipped vertically to match gridworld representation)
        arrow_dict = {
            0: (0, 0),      # stay / no movement
            1: (0, -0.4),   # up (flipped from original)
            2: (0, 0.4),    # down (flipped from original)
            3: (-0.4, 0),   # left
            4: (0.4, 0),    # right          
        }

        for s in range(self.solver.num_states):
            coord = self.solver.get_coordinates_from_state_index(s)
            row, col = coord
            # Use same coordinate system as value function plot
            y = row
            x = col

            # Choose best action(s) for deterministic policy
            best_actions = np.where(policy[s] == np.max(policy[s]))[0]
            for a in best_actions:
                if a == 0:
                    continue
                dx, dy = arrow_dict[a]
                ax.arrow(
                    x, y, dx, dy,
                    head_width=0.2, head_length=0.2, fc="k", ec="k"
                )

        ax.set_title(f"Policy for {self.env_name} - gamma={self.gamma}")
        
        # Invert y-axis to match value function plot coordinate system
        # ax.invert_yaxis()
        
        plt.tight_layout()

        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "..", "saved_figures")
            os.makedirs(figures_dir, exist_ok=True)
            subdir = os.path.join(figures_dir, f"{self.env_name}/{self._policy_type}")
            os.makedirs(subdir, exist_ok=True)
            filename = os.path.join(
                subdir, f"gamma={self.gamma}_policy.png"
            )

        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

    def _value_iteration(self) -> None:
        """
        Implement value iteration algorithm.
        
        This method iteratively updates state values based on the Bellman optimality
        equation until convergence. The student needs to implement:
        
        1. Value function update using the Bellman optimality equation:
           V(s) = max_a [ sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')] ]
           
        2. Policy update to be deterministic (probability 1 for best action):
           pi(s,a) = 1 if a = argmax_a Q(s,a), 0 otherwise
           where Q(s,a) = sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]
           
        3. Check for convergence by comparing old and new value functions
        
        The transition probabilities T and rewards R are pre-computed and stored in
        the T and R matrices respectively.
        """
        horizon = 50 #run policy evaluation for a fixed number of iterations
        unwrapped_env = self.env.unwrapped  # Get unwrapped environment

        v_i = np.zeros(unwrapped_env.num_states)
        p_i = np.zeros((unwrapped_env.num_states, unwrapped_env.num_actions))

        # Pre-compute transition and reward matrices
        T = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )
        R = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )

        for s in range(unwrapped_env.num_states):
            s_coord = self.solver.get_coordinates_from_state_index(s)
            for a in unwrapped_env.actions:
                next_state = self.solver.get_state_index_from_coordinates(
                    unwrapped_env.T(s_coord, a)[0][1]
                )
                T[s, a, next_state] = unwrapped_env.T(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )[0][0]
                R[s, a, next_state] = unwrapped_env.R(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )

        for _ in range(horizon):
            # Student code here     
            # Update value function
            delta = 0
            for s in range(unwrapped_env.num_states):
                v_i_old = v_i[s]
                max_value = -float('inf')
                for a in unwrapped_env.actions:
                    value = 0
                    for s_prime in range(unwrapped_env.num_states):
                        if T[s, a, s_prime] == 0:
                            continue
                        value += T[s, a, s_prime] * (R[s, a, s_prime] + self.gamma * v_i[s_prime])
                    if value > max_value:
                        max_value = value
                v_i[s] = max_value
                delta = max(delta, abs(v_i[s] - v_i_old))
            if delta < self.eps:
                break

        for s in range(unwrapped_env.num_states):
            best_action = 0
            max_value = -float('inf')
            for a in unwrapped_env.actions:
                value = 0
                for s_prime in range(unwrapped_env.num_states):
                    if T[s, a, s_prime] == 0:
                        continue
                    value += T[s, a, s_prime] * (R[s, a, s_prime] + self.gamma * v_i[s_prime])
                if value > max_value:
                    max_value = value
                    best_action = a
            p_i[s] = np.zeros(unwrapped_env.num_actions)
            p_i[s][best_action] = 1.0

        self.solver.set_policy_function(p_i)
        self.solver.set_value_function(v_i)

    def _deterministic_policy_iteration(self) -> None:
        """
        Implement deterministic policy iteration.
        
        This method alternates between policy evaluation and policy improvement steps,
        selecting the best action in each state deterministically. The student needs to implement:
        
        1. Policy Evaluation: Update value function using current policy:
           V(s) = sum_a pi(s,a) * sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]
           Note: Since policy is deterministic, this simplifies to:
           V(s) = sum_s' T(s,a*,s')[R(s,a*,s') + gamma * V(s')]
           where a* is the action with probability 1 in state s
           
        2. Policy Improvement: Update policy to be deterministic for best action:
           pi(s,a) = 1 if a = argmax_a Q(s,a), 0 otherwise
           where Q(s,a) = sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]
        
        The transition probabilities T and rewards R are pre-computed and stored in
        the T and R matrices respectively.
        """
        horizon = 50 # run policy evaluation for a fixed number of iterations
        unwrapped_env = self.env.unwrapped  # get unwrapped environment

        p_i = self.solver.get_policy_function()

        # Pre-compute transition and reward matrices
        T = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )
        R = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )

        for s in range(unwrapped_env.num_states):
            s_coord = self.solver.get_coordinates_from_state_index(s)
            for a in unwrapped_env.actions:
                next_state = self.solver.get_state_index_from_coordinates(
                    unwrapped_env.T(s_coord, a)[0][1]
                )
                T[s, a, next_state] = unwrapped_env.T(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )[0][0]
                R[s, a, next_state] = unwrapped_env.R(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )

        v_i = np.zeros(unwrapped_env.num_states)
        for k in range(50):
            print("Policy Iteration %d" % k)

            # Policy Evaluation
            elapsed = time.time()

            for _ in range(horizon):
                # Get expected value for current policy
                for s in range(unwrapped_env.num_states):
                    a = max(enumerate(p_i[s]), key=lambda x: x[1])[0]
                    value = 0
                    for s_prime in range(unwrapped_env.num_states):
                        if T[s, a, s_prime] == 0:
                            continue
                        value += T[s, a, s_prime] * (R[s, a, s_prime] + self.gamma * v_i[s_prime])
                    v_i[s] = value

            elapsed = time.time() - elapsed
            print(".....Evaluate done in %g" % elapsed)
            elapsed = time.time()

            # Policy Improvement
            stable = True
            p_i_new = np.zeros_like(p_i)  # Create new policy
            for s in range(unwrapped_env.num_states):
                a_old = np.argmax(p_i[s])  # Get the action with highest probability
                max_value = -float('inf')
                best_action = 0
                for a in range(unwrapped_env.num_actions):
                    value = 0
                    for s_prime in range(unwrapped_env.num_states):
                        value += T[s, a, s_prime] * (R[s, a, s_prime] + self.gamma * v_i[s_prime])
                    if max_value < value:
                        max_value = value
                        best_action = a
                
                # Create deterministic policy: probability 1 for best action, 0 for others
                p_i_new[s] = np.zeros(unwrapped_env.num_actions)
                p_i_new[s][best_action] = 1.0
                
                if best_action != a_old:
                    stable = False
            
            p_i = p_i_new  # Update policy

            elapsed = time.time() - elapsed
            print(".....Improve done in %g" % elapsed)

            self.solver.set_policy_function(p_i)
            self.performance_history.append(self.solve(max_steps=20)[0])
            if stable:
                print("Policy is stable. Stopping policy iteration.")
                break

        self.solver.set_value_function(v_i)


if __name__ == "__main__":

    ############ Q1.1 ############
    gw0_pi_solver = GridworldSolver(policy_type="pi", gridworld_map_number=0)
    gw1_pi_solver = GridworldSolver(policy_type="pi", gridworld_map_number=1)

    for solver in [gw0_pi_solver, gw1_pi_solver]:
        for gamma in [0.99, 0.9, 0.75, 0.5]:
            solver.gamma = gamma
            start_time = time.time()
            solver.compute_policy()
            elapsed_time = time.time() - start_time
            print("Computed Q2 PI Policy in %g seconds" % elapsed_time)
            solver.plot_policy_curve(solver.performance_history)
            _, fig = solver.plot_value_function(solver.solver.get_value_function())
            solver.plot_policy(solver.solver.get_policy_function(), ax=fig.axes[0])


    ############ Q1.2 ############
    gw0_vi_solver = GridworldSolver(policy_type="vi", gridworld_map_number=0)
    gw1_vi_solver = GridworldSolver(policy_type="vi", gridworld_map_number=1)
    for solver in [gw0_vi_solver, gw1_vi_solver]:
        for gamma in [0.99, 0.9, 0.75, 0.5]:
            solver.gamma = gamma
            start_time = time.time()
            solver.compute_policy()
            elapsed_time = time.time() - start_time
            print("Computed Q1.a VI Policy in %g seconds" % elapsed_time)
            _, fig = solver.plot_value_function(solver.solver.get_value_function())
            solver.plot_policy(solver.solver.get_policy_function(), ax=fig.axes[0])
