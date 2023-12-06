import numpy as np

# Enum like for action_choice types
class ActionChoiceType:
    GREEDY = 0
    EPSILON_GREEDY = 1

# Function to choose an action based on epsilon-greedy policy
def choose_action(state, action_choice_type):
    if action_choice_type == ActionChoiceType.GREEDY:
        max_val = np.max(Q_table[state, :])
        is_tie = np.any(Q_table[state, :] == max_val)
        if is_tie:
            return 0
        else:
            return np.argmax(Q_table[state, :])
    elif action_choice_type == ActionChoiceType.EPSILON_GREEDY:
        if np.random.rand() < (1 - EPSILON): # Choose greedy action with probability 1 - epsilon
            return np.argmax(Q_table[state, :])  # Exploitation
        else:
            return np.random.randint(NUM_ACTIONS)  # Exploration - uniformly choose between 0 and 1

# Function to perform Q-learning update
def q_learning_update(state, action, reward, next_state):
    max_next_q_value = np.max(Q_table[next_state, :])
    Q_table[state, action] = (1 - ALPHA) * Q_table[state, action] + ALPHA * (reward + GAMMA * max_next_q_value)

def train(action_choice_type):
    for episode in range(NUM_EPISODES):
        # Start in state s1
        current_state = 0
        
        for _ in range(NUM_STEPS):
            # Choose action based on behavior policy
            chosen_action = choose_action(current_state, action_choice_type)
            
            # Perform the chosen action
            if chosen_action == 0:  # Move
                next_state = 1 - current_state  # Move to the other state
                reward = 0  # Reward for move
            else:  # Stay
                next_state = current_state  # Stay in the current state
                reward = 1  # Reward for stay
            
            # Update Q-value
            q_learning_update(current_state, chosen_action, reward, next_state)
            
            # Move to the next state
            current_state = next_state
            
if __name__ == "__main__":
    # Constants
    NUM_STATES = 2
    NUM_ACTIONS = 2
    ALPHA = 0.5  # Step size parameter
    GAMMA = 0.8  # Discounting factor
    EPSILON = 0.1  # Exploration probability
    NUM_STEPS = 200 # Maximum number of steps per episode
    NUM_EPISODES = 1 # Number of episodes

    # Part 1 - use greeedy action
    # Initialize Q-table with zeros
    Q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
    train(ActionChoiceType.GREEDY)
    print(f"Q-table after {NUM_EPISODES} episodes with greedy action:")
    print(Q_table)

    # Part 2 - use epsilon-greedy action
    EPSILON = 0.5
    Q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
    train(ActionChoiceType.EPSILON_GREEDY)
    print(f"Q-table after {NUM_EPISODES} episodes with epsilon-greedy action:")
    print(Q_table)