# CartPole with Custom Interval Rewards using Stable Baselines3

## Overview

This project explores training a Reinforcement Learning (RL) agent to master the CartPole-v1 environment from Gymnasium, with a focus on achieving extended balancing times. The core of this project is a custom reward wrapper, `IntervalSurvivalRewardWrapper`, designed to provide more structured feedback to the agent compared to the default environment reward. The agent is trained using the Proximal Policy Optimization (PPO) algorithm from the Stable Baselines3 library.

This project served as a hands-on learning experience in Python, RL fundamentals, environment customization, hyperparameter tuning, and experiment management.

## Key Features

* **Custom Reward Shaping:** Implements an `IntervalSurvivalRewardWrapper` that provides:
    * A periodic reward for every N seconds the pole is balanced.
    * A small, constant per-step survival bonus to encourage continued effort.
* **Reinforcement Learning Agent:** Uses the PPO algorithm from Stable Baselines3.
* **Experiment Management:** Demonstrates basic setup for logging training progress with TensorBoard and saving/loading models.
* **Organized Code Structure:** Includes helper utilities and a modular environment wrapper.

## Motivation & Learning Journey

The primary motivation was to learn practical Reinforcement Learning by tackling a classic control problem and exploring how different reward structures impact agent learning.

Initially, the agent struggled to learn effectively or got stuck at local optima (e.g., consistently failing around the 10-10.5 second mark, just before or after the first interval reward). This led to several iterations and learnings:

1.  **Environment Limits:** Ensured `max_episode_steps` for the CartPole environment was sufficiently increased to allow for longer survival times.
2.  **Reward Sparsity:** Moved from potentially very sparse rewards to more frequent interval-based rewards to provide clearer learning signals.
3.  **Exploration vs. Exploitation:** Tuned hyperparameters like `ent_coef` (entropy coefficient) to encourage more exploration, helping the agent discover strategies beyond initial local optima.
4.  **Future Reward Valuation:** Adjusted `gamma` (discount factor) to ensure the agent appropriately valued longer-term survival and the prospect of future interval rewards.
5.  **Path Handling & Refactoring:** Improved code structure for clarity and robust path management using `pathlib`.

This project demonstrates an iterative approach to solving RL problems.

## Technologies Used

* **Python 3.9+**
* **Gymnasium:** For the CartPole environment and wrapper base class.
* **Stable Baselines3:** For the PPO reinforcement learning algorithm.
* **NumPy:** For numerical operations (often used by RL libraries).
* **TensorBoard:** For visualizing training progress.
* **Tqdm:** For progress bars during evaluation.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Make sure you have a `requirements.txt` file in your repository. You can generate it using:
    ```bash
    pip freeze > requirements.txt
    ```
    Then, install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    (Key dependencies to include in `requirements.txt` would be `gymnasium`, `stable-baselines3[extra]`, `numpy`, `tqdm`, `tensorboard`).

## How to Run

The main script for training and evaluation is `main.py`.

### Training

To train a new agent (or continue training if a saved model exists and loading is enabled in `main.py`):

```bash
python main.py
