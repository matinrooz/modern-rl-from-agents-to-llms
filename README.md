# modern-rl-from-agents-to-llms

The exercise notebooks of the course *Modern Reinforcement Learning: From Agents to LLMs*
(Prof. Ilija Bogunovic, University of Basel, spring 2026,
[course page](https://rhinelab.ai/teaching/modern-rl-spring-2026)). They go from dynamic
programming on a small grid to RLHF, DPO and GRPO on small language models.

Notebooks 1 to 7 are my solutions. Notebooks 8 and 9 are the solutions we worked through
together in the exercise session, the code there is from the course (see the note at the top
of each notebook).

| notebook | topic | what is in it |
|---|---|---|
| `project1.ipynb` | MDPs and dynamic programming | value iteration and policy iteration on a 6×6 grid with a goal, a trap and slippery moves |
| `project2.ipynb` | Monte Carlo and TD control | first-visit vs. last-visit MC, SARSA and Q-learning on a GridWorld |
| `project3.ipynb` | function approximation | semi-gradient TD(0) with state aggregation on a 100-state random walk, DQN on CartPole |
| `project4.ipynb` | policy gradients | REINFORCE, REINFORCE with a learned baseline, actor-critic on CartPole |
| `project5.ipynb` | GAE, TRPO, PPO | λ sweep for GAE, TRPO and PPO with the clipped objective on CartPole |
| `project6.ipynb` | imitation learning | behavior cloning and DAgger on a custom 2D task (follow an S-shaped curve) |
| `project7.ipynb` | preference learning | Bradley–Terry reward model on a 2D toy problem, and reward hacking |
| `project8.ipynb` | RLHF: PPO-KL and DPO | GPT-2 + LoRA, SFT on IMDB reviews, then PPO with a KL penalty and DPO, with a DistilBERT sentiment classifier as reward |
| `project9.ipynb` | GRPO | Qwen3-0.6B + LoRA trained with GRPO on Countdown (reach a target number with +, −, ×, ÷), verifiable reward, MAD normalisation, clipping, temperature |

## Results

All notebooks except 3 and 9 are saved with their outputs and plots (CPU / Apple M1).

**Notebook 1:** value iteration converges after 1376 sweeps, policy iteration after 5 policy
updates. With the greedy policy from value iteration, 50 out of 50 rollouts reach the goal,
average return 0.49 (each step costs 0.04).

**Notebook 2:** bias and variance of the estimate of one Q-value over 25 runs (true value
about −2.12):

| method | bias | variance |
|---|---|---|
| MC, last visit | −9.57 | 54.9 |
| MC, first visit | −7.74 | 19.6 |
| TD (Q-learning) | −0.47 | 0.02 |

**Notebook 4:** a baseline reduces the variance of the policy gradient by 2.9× in the toy
example. On CartPole, REINFORCE with normalised returns and REINFORCE with a learned baseline
both reach a mean return of about 490 (max 500) over the last 50 episodes; the baseline
version learns faster (409 vs. 297 after 200 episodes). The one-step actor-critic did not
learn in this run (mean return around 20).

**Notebook 5:** GAE λ sweep, TRPO and PPO on CartPole, the results are in the plots.

**Notebook 6:** with noisy execution, behavior cloning drops to a mean reward of −207, while the
expert gets −56.5. DAgger brings it back to −57.4, which closes 99% of the gap.

**Notebook 7:** the Bradley–Terry reward model learned from pairwise preferences has a
correlation of 0.997 with the true reward on the training distribution and 0.934 out of
distribution.

**Notebook 8:** mean sentiment reward of the generated reviews (from −1 to +1):

| model | reward |
|---|---|
| base GPT-2 | +0.44 |
| SFT | +0.25 |
| PPO-KL | +0.50 |
| DPO | +0.99 |

DPO gave the most positive reviews. PPO-KL was unstable: the reward went up to +0.77 after 60
iterations and then dropped again while the KL to the reference grew to 2.2, and some
samples with high reward were not real sentences anymore (reward hacking).

## State

- The DQN part of notebook 3 is not finished: the Q-network, the replay buffer sampling and
  the update step are still TODOs, so notebook 3 is saved without outputs.
- Notebook 9 is saved without outputs. It needs a GPU with about 16 GB (it was made for a
  Kaggle P100).

## Run

```bash
pip install torch numpy matplotlib tqdm gymnasium transformers peft
jupyter notebook
```

Notebook 9 also needs `reasoning-gym` and the Countdown data from the course.

## References

- R. S. Sutton and A. G. Barto. *Reinforcement Learning: An Introduction*, 2nd ed., 2018
- J. Schulman et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017
- S. Ross, G. Gordon and D. Bagnell. A Reduction of Imitation Learning and Structured
  Prediction to No-Regret Online Learning (DAgger). AISTATS 2011
- P. Christiano et al. Deep Reinforcement Learning from Human Preferences. NeurIPS 2017
- L. Ouyang et al. Training Language Models to Follow Instructions with Human Feedback.
  NeurIPS 2022
- R. Rafailov et al. Direct Preference Optimization: Your Language Model is Secretly a Reward
  Model. NeurIPS 2023
- Z. Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
  Models (GRPO). arXiv:2402.03300, 2024
- E. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022
