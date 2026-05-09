# DQN — CarRacing-v3

DQN agent trained on the Gymnasium `CarRacing-v3` environment with a discrete action space.

## Requirements

- Train at least one DRL model : DQN ✅
- Have at least one experiment to evaluate the performance of your model in. ❌
    - Evaluating returns obtained by agent during or after training is fine
    - Lets evaluate different environments or algorithms
    - Maybe try to experiment on what environments DQN performs better
- Do a non-trivial amount of programming work. ❌
    - Cite the work you used (libraries, etc.) 
    - Demonstrate a sufficient level of understanding and ability to meaningfully interact with, dissect, and modify the code you start out with. This can be done, for example, by implementing original and new evaluation criteria or other interesting statistics for your experiment, trying to add new extensions to a standard training algorithm, or correctly implementing ablations by carefully removing specific components of an algorithm whilst leaving the general idea of the base algorithm intact.
- Record a video presentation, of at most 10 minutes.

## How to get a higher grade
Things I am gonna try to achieve a higer grade:
- Solid hyperparameter tuning, analysis of interactions between hyperparameters or sensitivity of learning to hyperparameters.
- Use advanced algorithms and/or evaluation methodologies that go beyond the basics covered in the lectures.
- Explain technical details of your algorithms and experimental setup particularly well and clearly (beyond a minimum sufficient explanation).
- Particularly effective presentation and visualisation of results (beyond a simple, single figure of e.g. returns over training time). Tools like Weights&Biases or Tensorboard can help here. 

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Train:**
```bash
python3 -m src.train
```
Checkpoints are saved every 10 episodes as `dqn_ep<n>.pt`, and a final model as `dqn_final.pt`.

**Evaluate:**
```bash
python3 -m src.evaluate --model src/weights/dqn_ep<n>.pt --render
```

## ToDo
todo rl final assignment:
1. understand the current implementation
2. think about an experiment based on the assignment requirements, so you have a goal to work to.
3. only after that start training. 

## Ideas for experiments
- grayscale vs color
- Hyperparameter sensitivity (TARGET_UPDATE, BUFFER_SIZE)
- Architecture ablatian

