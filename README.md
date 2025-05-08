# POWR: Policy Mirror Descent with Operator World-Models for Reinforcement Learning

## How to Run Experiments

### Clone the Repository
(in your command line interface)
```bash
git clone https://github.com/EdwinRou/powr.git
```

### Set Up the Environment
1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv powr_env
   source powr_env/bin/activate  # Linux/MacOS
   powr_env\Scripts\activate   # Windows
   ```

2. **Navigate to the project folder:**
   ```bash
   cd powr
   ```

3. **Install the required Python dependencies:**
   ```bash
   pip install -r requirements_experiment.txt
   ```

4. **Install OpenCV dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install libgl1-mesa-glx
   ```

### Set Up Weights & Biases (Wandb)
1. **Create a Wandb account:**
   - Visit [wandb.ai](https://wandb.ai) and sign up for an account. Academic accounts are free and offer additional benefits.

2. **Update `wandb.init` in `baseline_cartpole.py`:**
   - Modify the `entity` field to match your Wandb username or username+institution (for academic accounts).
3. run a wandb login and enter your api key, available at : ?

### Experiment Details
#### Baseline Experiment
1. **Environment:**
   - `CartPole` from [Gymnasium](https://gymnasium.farama.org/).
2. **Algorithm:**
   - Deep Q-Network (DQN).
3. **Configuration:**
   - Modify the `config` dictionary in `baseline_cartpole.py` to adjust the parameters (not recommanded initially).

4. **Run the baseline experiment:**
   ```bash
   python baseline_cartpole.py
   ```

#### GPU Considerations
- Ensure `cuda` is set up to use a GPU:
  ```python
  model = DQN(args, device="cuda")
  ```
- If CUDA is unavailable or not configured, remove `device="cuda"` to use the CPU instead.

### Run the POWR Experiment
#### Example Command
```bash
python train.py \
--project operator_learning_Taxi \
--env Taxi-v3 \
--eta 0.1 \
--gamma 0.99 \
--la 1e-6 \
--sigma 0.2 \
--subsamples 1_000 \
--q-mem 0 \
--warmup-episodes 12 \
--train-episodes 6 \
--eval-episodes 3 \
--iter-pmd 2 \
--epochs 20 \
--device gpu \
```

#### Save GIF Visualizations
- Add the following argument to save environment visuals:
  ```bash
  --save-gif-every 1
  ```
  - Visualizations will be saved to Wandb for easy access.

#### GPU Requirement
- A GPU is required for acceptable runtime performance.
  ```bash
  --device gpu
  ```

## Potential Issues and Solutions
1. **CUDA Compatibility:**
   - Ensure CUDA is installed and correctly set up. If issues arise, switch to CPU mode as described above.

2. **Environment Dependencies:**
   - Ensure OpenCV and other dependencies are properly installed.

---

This guide provides a structured approach to setting up and running experiments for the POWR framework. For further details, refer to the [POWR paper](https://github.com/CSML-IIT-UCL/powr).




# Original Documentation :
# POWR: Operator World Models for Reinforcement Learning

[Paper](https://arxiv.org/pdf/2406.19861) / [Website](https://csml-iit-ucl.github.io/powr/)

##### [Pietro Novelli](https://scholar.google.com/citations?user=bXlwJucAAAAJ&hl=en), [Marco Pratticò](https://scholar.google.com/citations?user=gC9M9AkAAAAJ&hl=en&oi=ao), [Massimiliano Pontil](https://scholar.google.com/citations?user=lcOacs8AAAAJ&hl=it) ,[Carlo Ciliberto](https://scholar.google.com/citations?user=XUcUAisAAAAJ&hl=it)

This repository contains the code for the paper **"Operator World Models for Reinforcement Learning"**.

*Abstract:* Policy Mirror Descent (PMD) is a powerful and theoretically sound methodology for sequential decision-making. However, it is not directly applicable to Reinforcement Learning (RL) due to the inaccessibility of explicit action-value functions. We address this challenge by introducing a novel approach based on learning a world model of the environment using conditional mean embeddings (CME). We then leverage the operatorial formulation of RL to express the action-value function in terms of this quantity in closed form via matrix operations. Combining these estimators with PMD leads to POWR, a new RL algorithm for which we prove convergence rates to the global optimum. Preliminary experiments in both finite and infinite state settings support the effectiveness of our method, making this the first concrete implementation of PMD in RL to our knowledge.


Our release is **under construction**, you can track its progress below:

- [x] Installation instructions
- [ ] Code implementation
	- [x] Training
	- [x] Testing
	- [x] Optimization
	- [x] Model saving and loading
	- [ ] Cleaning
- [ ] Reproducing paper results scripts
- [ ] Hyperparameters for each env
- [ ] Trained models
- [ ] Complete the README

## Installation

1. Install POWR dependencies:
```
conda create -n powr python=3.11
conda activate powr 
pip install -r requirements.txt
```

2. (optional) set up `wandb login` with your WeightsAndBiases account. If you do not wish to use wandb to track the experiment results, run it offline adding the following arg `--offline`. For example, `python3 train.py --offline`

## Getting started

### Quick test
- `python3 train.py`

## Cite us
If you use this repository, please consider citing
```
@misc{novelli2024operatorworldmodelsreinforcement,
      title={Operator World Models for Reinforcement Learning}, 
      author={Pietro Novelli and Marco Pratticò and Massimiliano Pontil and Carlo Ciliberto},
      year={2024},
      eprint={2406.19861},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.19861}, 
}
```
