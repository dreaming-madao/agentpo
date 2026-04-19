# AgentPO

**AgentPO: Enhancing Multi-Agent Collaboration via Reinforcement Learning**

This repository contains the official implementation of *AgentPO* (**ICLR 2026**). It trains a dedicated **Collaborator** agent with reinforcement learning under a fixed multi-agent topology, optimizing collaboration with an **Actor** agent to improve end-to-end performance on reasoning tasks.

**Paper:** *AgentPO: Enhancing Multi-Agent Collaboration via Reinforcement Learning* — code: [github.com/sunlin-ai/agentpo](https://github.com/sunlin-ai/agentpo).

## Repository layout

```
.
├── agentpo/                 # Core training, rewards, and related logic
│   ├── main_dapo.py         # Training entry point (DAPO / verl)
│   ├── reward_manager_agentpo.py
│   ├── rl_dataset.py        # Dataset pipeline (referenced by scripts)
│   └── evaluation/          # Math benchmarks (includes latex2sympy, etc.)
├── scripts/
│   ├── train.sh             # Example training script (edit paths & hyperparameters)
│   └── test.sh              # Example evaluation script (edit paths & checkpoints)
└── verl/                    # Upstream [verl](https://github.com/volcengine/verl) (editable install)
```

The evaluation pipeline is adapted from [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness). Training builds on the verl stack.

---

## Setup

1. **Python**: Python 3.10+ recommended, with a CUDA-enabled PyTorch build.

2. **Install verl (required)** from the repository root:

   ```bash
   cd verl
   pip install -e .
   ```

   See the verl documentation for optional components (e.g., vLLM, FSDP).

3. **Add the repository root to `PYTHONPATH`** so `python -m agentpo.main_dapo` resolves the `agentpo` package:

   ```bash
   export PYTHONPATH="/path/to/this/repo:${PYTHONPATH}"
   ```

4. **Extra dependencies for evaluation** (when running scripts under `agentpo/evaluation`): see [`agentpo/evaluation/README.md`](agentpo/evaluation/README.md) for local `latex2sympy` install and `requirements.txt`.

---

## Training

Example (matches the bundled script; **edit model paths, data paths, `HOME`, etc. in `scripts/train.sh` for your environment**):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh
```

The main entry point is `python -m agentpo.main_dapo` with Hydra configuration; `train.sh` illustrates typical settings for the AgentPO reward manager and cooperation mode. For algorithm and implementation details, refer to the paper (Section 2 and appendices).

---

## Evaluation

Example (**update `scripts/test.sh` for your checkpoint, `MERGE_MODEL_PATH`, datasets, and GPUs**):

```bash
bash scripts/test.sh
```

The script invokes `agentpo/evaluation/math_eval.py` with multi-dataset and vLLM options; see [`agentpo/evaluation/README.md`](agentpo/evaluation/README.md).

---

## Citation

If you use this code or the paper, please cite:

```bibtex
@inproceedings{sun2026agentpo,
  title     = {AgentPO: Enhancing Multi-Agent Collaboration via Reinforcement Learning},
  author    = {Sun, Lin and Liu, Chuang and Zhang, Can and Wu, Yubin and Lu, Weijia and Wu, Ning},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```

---

## Acknowledgements

- Training stack: [verl](https://github.com/volcengine/verl) (Apache License 2.0).
- Math evaluation: [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness).
