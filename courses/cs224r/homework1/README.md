# Homework 1, Problem 1: behavioral cloning

Stanford CS224R imitation-learning homework, studied independently. This folder is the official starter. Problem 1 is behavior cloning with MSE on the Flappy Bird environment. Flow matching and DAgger stay in the starter and are not started yet.

## Problem 1

Expert demonstrations are state-action pairs. `BCPolicy` is a 3-layer MLP that maps the 4-D observation to an action chunk of length 20. `mse_loss` trains it to match the expert chunk. At rollout only the first 10 actions are executed.

Easy mode has one gap, so the expert action is nearly a function of the state. Hard mode has two gaps, so the same state can have two expert actions. MSE fits the mean of those actions. The writeup asks for both runs and a short explanation of the hard-mode result.

## Environment

`deep-rl-hw1`, Python 3.10, CPU PyTorch. Created from `environment.yml` in this folder.

```powershell
conda activate deep-rl-hw1
$env:SDL_VIDEODRIVER = "dummy"
$env:SDL_AUDIODRIVER = "dummy"
python main.py --method bc_reg --env easy
python main.py --method bc_reg --env hard