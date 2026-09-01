# Flappy Bird Reinforcement Learning Agents
> Aprendizaje Automático II  
> Tecnicatura Universitaria en Inteligencia Artificial (Universidad Nacional de Rosario)  
> Máximo Alva, María Sol Aranda  
> 2025  

This project explores **Reinforcement Learning** by training agents to play **Flappy Bird** using two different approaches:

- **Tabular Q-Learning** with a discretized state space.
- **Neural Network approximation of the learned Q-function.**

The project focuses on comparing the limitations of a traditional tabular approach with a neural network capable of approximating the knowledge learned by the Q-table.

## Overview

The Flappy Bird environment contains a continuous state space, where variables such as the player's position, vertical velocity, and the relative position of the next pipes can take many different values.

To make the problem suitable for tabular Q-Learning, the continuous environment state was transformed into a finite discrete state space.

The project follows this workflow:

1. Extract relevant variables from the game state.
2. Discretize the continuous state space.
3. Train a **Q-Learning agent** to learn a policy.
4. Store the learned Q-values in a Q-table.
5. Use the learned Q-table as a dataset to train a **Neural Network**.
6. Compare the performance of both approaches.
   
## Agent 1 — Tabular Q-Learning

The first agent uses **Q-Learning** with a manually discretized state space.

### State Engineering

The original Flappy Bird environment is continuous, making it impractical to directly represent every possible state in a Q-table.

Relevant state variables are therefore divided into discrete intervals (**bins**). Each continuous value is mapped to an index, producing a finite representation of the game state.

This allows every state-action combination to be associated with a Q-value.

### State Discretization

An important part of the project was tuning the discretization process.

Initially, the state space was too coarse, causing many different game situations to be mapped to the same discrete state. As a result, the agent could not reliably distinguish situations where it should jump from those where it should not.

The discretization was refined by increasing the number of bins, allowing the agent to represent the environment with greater precision.

State values were also bounded to ensure that observations remain within the trained state space and to prevent extreme values from generating unseen states.

### Results

The final Q-Learning agent achieved the following results over **30 episodes**:

| Metric | Result |
| :--- | :--- |
| Average Score | 95.63 |
| Maximum Score | 278 |
| Average Survival | 3,645 steps |
| Average Reward | 90.6 |

The agent successfully learned a reasonably effective policy, although its performance is naturally limited by the discretization of the state space.

## Agent 2 — Neural Network Q-Function Approximation

The second approach uses a **Neural Network** trained to approximate the Q-values learned by the tabular agent.

Instead of directly storing Q-values for every discrete state, the neural network learns the relationship between game states and their corresponding action values.

The trained model is then used by a neural agent to select the action with the highest predicted Q-value.

### Initial Approach

The first neural network was trained using the complete Q-table.

However, this introduced a major problem: a large portion of the table contained states `Q = [0, 0]` that had never been visited during Q-Learning.

As a consequence, the neural network was trained on a large amount of data containing no useful information.

This created a strong bias toward predicting values close to zero and prevented the model from correctly approximating the learned policy.

### Initial Neural Network Results

Over **30 episodes**, the first version achieved:

| Metric | Result |
| :--- | :--- |
| Average Score	| 14.13 |
| Maximum Score	| 41 |
| Average Survival | 571 steps |
| Average Reward | 9.1 |

The results showed that the model was not successfully reproducing the policy learned by the Q-Learning agent.

### Improved Neural Network

The training dataset was then modified to remove states that had never been visited during Q-Learning.

Only states containing meaningful learned Q-values were used to train the model.

By removing the large number of irrelevant `Q = [0, 0]` states, the neural network was able to focus on the actual knowledge acquired by the Q-Learning agent.

### Final Results

The improved neural agent achieved:

| Metric | Result |
| :--- | :--- |
| Average Score | 680.3 |
| Maximum Score | 1,423 |
| Average Survival | 25,684 steps |
| Average Reward | 675.3 |

These results were obtained over **10 episodes** due to execution time limitations.

Despite the smaller number of evaluation episodes, the performance improvement was substantial.

## Results Comparison
|Agent | Average Score | Maximum Score | Average Survival |
| :--- | :--- | :--- | :--- |
|Q-Learning | 95.63 | 278 | 3,645 |
|Neural Network — Initial | 14.13 | 41 | 571 |
|Neural Network — Improved | 680.3 | 1,423 | 25,684 |

The final neural network significantly outperformed the tabular Q-Learning agent.

The main improvement came from training the model exclusively on meaningful states instead of including the large number of unvisited states with zero Q-values.

## Technologies
- Python
- Reinforcement Learning
- Q-Learning
- Neural Networks
- TensorFlow / Keras
- NumPy
- PLE (PyGame Learning Environment)
- Flappy Bird
  
## Project Structure

```
.
├── agents/                         # Agent implementations
├── ple/                            # Game environment
│
├── train_q_agent.py                # Q-Learning training
├── train_q_nn.py                   # Neural Network training
├── test_agent.py                   # Main script for testing agents in Flappy Bird.
│
├── flappy_birds_q_table.pkl        # Trained Q-table
├── flappy_birds_q_table_final.pkl  # Final trained Q-table
├── flappy_q_nn_model.keras         # Trained Neural Network
├── state_norm.npy                  # State normalization data
│
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/maximoalva/flappy-bird-reinforcement-learning-agents.git
cd flappy-bird-reinforcement-learning-agents
```

Create and activate a virtual environment:

```bash
python -m venv env
```

On Linux/macOS:

```bash
source env/bin/activate
```

On Windows:

```bash
env\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Train the Agents

#### Q-Learning Agent

```bash
python train_q_agent.py
```

#### Neural Network Agent

Train the Neural Network using the learned Q-table:

```bash
python train_q_nn.py
```

### Test the Agents

The project includes different agents that can be executed through `test_agent.py`.

#### Random Agent

Agent that takes random actions.

```bash
python test_agent.py --agent agentes.random_agent.RandomAgent
```

#### Manual Agent

Play the game manually using the **Space bar**.

```bash
python test_agent.py --agent agentes.manual_agent.ManualAgent
```

#### Q-Learning Agent

```bash
python test_agent.py --agent agentes.dq_agent.QAgent
```

#### Neural Network Agent

```bash
python test_agent.py --agent agentes.nn_agent.NNAgent
```

## Key Takeaways

This project highlights several important aspects of Reinforcement Learning and function approximation:

- **State representation is critical** for tabular reinforcement learning.
- Coarse discretization can cause important game situations to become indistinguishable.
- Q-tables can be used as datasets to train models that approximate learned value functions.
- Training data quality is crucial when using neural networks.
- Including large amounts of uninformative data can significantly bias a model.
- After filtering unvisited states, the neural network was able to successfully approximate—and significantly outperform—the tabular Q-Learning policy.
