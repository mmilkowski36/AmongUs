<<<<<<< HEAD
# *AmongUs*: A Sandbox for Agentic Deception

This project introduces the game "Among Us" as a model organism for lying and deception and studies how AI agents learn to express lying and deception, while evaluating the effectiveness of AI safety techniques to detect and control out-of-distribution deception.

## Overview

The aim is to simulate the popular multiplayer game "Among Us" using AI agents and analyze their behavior, particularly their ability to deceive and lie, which is central to the game's mechanics.

<img src="https://static.wikia.nocookie.net/among-us-wiki/images/f/f5/Among_Us_space_key_art_redesign.png" alt="Among Us" width="400"/>

## Setup

1. Clone the repository:
   ```bash
   git clone XXXX
   cd AmongUs
   ```

2. Set up the environment:
   ```bash
   conda create -n amongus python=3.10
   conda activate amongus
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run Games

To run the sandbox and log games of various LLMs playing against each other, run:

```
main.py
```
You will need to add a `.env` file with an [OpenRouter](https://openrouter.ai/) API key.
Or, you can run using only a local llamam - configure in among-agents/amongagents/envs/configs/experiment_config.py

## Deception ELO

To reproduce our Deception ELO and Win Rate results, run:

```
python elo/deception_elo.py
```

## Caching Activations

Once the (full) game logs are in place, use the following command to cache the activations of the LLMs:

```
python linear-probes/cache_activations.py --dataset <dataset_name>
```

This loads up the HuggingFace models and caches the activations of the specified layers for each game action step. This step is computationally expensive, so it is recommended to run this using GPUs.

Use `configs.py` to specify the model and layer to cache, and other configuration options.

## LLM-based Evaluation (for Lying, Awareness, Deception, and Planning)

To evaluate the game actions by passing agent outputs to an LLM, run:

```
bash evaluations/run_evals.sh
```
You will need to add a `.env` file with an OpenAI API key.


(TODO)

## Training Linear Probes

Once the activations are cached, training linear probes is easy. Just run:

```
python linear-probes/train_all_probes.py
```
You can choose which datasets to train probes on - by default, it will train on all datasets.

## Evaluating Linear Probes

To evaluate the linear probes, run:

```
python linear-probes/eval_all_probes.py
```
You can choose which datasets to evaluate probes on - by default, it will evaluate on all datasets.

It will store the results in `linear-probes/results/`, which are used to generate the plots in the paper.

## Sparse Autoencoders (SAEs)

We use the [Goodfire API](https://goodfire.ai/) to evaluate SAE features on the game logs. To do this, run the notebook:

```
reports/2025_02_27_sparse_autoencoders.ipynb
```
You will need to add a `.env` file with a Goodfire API key.

## Project Structure

```plaintext
.
├── CONTRIBUTING.md         # Contribution guidelines
├── Dockerfile               # Docker setup for project environment
├── LICENSE                  # License information
├── README.md                # Project documentation (this file)
├── among-agents             # Main code for the Among Us agents
│   ├── README.md            # Documentation for agent implementation
│   ├── amongagents          # Core agent and environment modules
│   ├── envs                 # Game environment and configurations
│   ├── evaluation           # Evaluation scripts for agent performance
│   ├── notebooks            # Jupyter notebooks for running experiments
│   ├── requirements.txt     # Python dependencies for agents
│   └── setup.py             # Setup script for agent package
├── expt-logs                # Experiment logs
├── k8s                      # Kubernetes configurations for deployment
├── main.py                  # Main entry point for running the game
├── notebooks                # Additional notebooks (not part of the main project)
├── reports                  # Experiment reports
├── requirements.txt         # Python dependencies for main project
├── tests                    # Unit tests for project functionality
└── utils.py                 # Utility functions
```


=======
# AmongUs Server

This repository contains the human-trials FastAPI game server and the
`among-agents` game engine package it depends on.

## Local Development

```bash
python -m venv venv
source venv/bin/activate
make install-dev
make install-browser
make run
```

Open `http://127.0.0.1:8011`.

## LLM Provider

The server calls model providers directly. Configure one provider in `.env`:

```bash
LLM_PROVIDER=gemini  # openai, gemini, or anthropic
LLM_MODEL=gemini-3.5-flash
GEMINI_API_KEY=...
```

Use `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY` for the selected
provider. Optional role-specific overrides are also supported:
`CREWMATE_LLM_MODEL`, `IMPOSTOR_LLM_MODEL`, `CREWMATE_LLM_MODELS`, and
`IMPOSTOR_LLM_MODELS`.

For headless browser checks, Playwright may require OS packages. Check them with:

```bash
make check-browser-deps
```

If packages are missing, run this manually in an interactive terminal so sudo can
prompt:

```bash
venv/bin/python -m playwright install-deps chromium
```

Then run:

```bash
make check-matchmaking
```

The stable ASGI app import is:

```text
amongus_server.main:app
```

See `DEPLOYMENT.md` for the dsg7 Apache/systemd shape.

## Matchmaking Quotas

Five-player matchmaking targets 100 completed games of each composition, from
one human/four AI through five humans/no AI. Configure these settings in `.env`
and restart the server:

```dotenv
MATCHMAKING_QUOTA_PER_CONFIGURATION=100
MATCHMAKING_QUOTA_START_DATE=2026-09-07
```

The cutoff includes midnight on that date in America/New_York. Counts use the
original roster and a recorded Crewmates or Impostors winner in the live
`EXPERIMENT_PATH/game_data.db` (default: `human_trials/logs/game_data.db`).
Database snapshots elsewhere in the repository are not included automatically.

Lobbies immediately fill the AI seats required by the largest human count still
needed. For example, once five-human games reach quota, a new lobby starts with
one human and one AI, leaving three seats for humans. Further arrivals beyond
the human limit enter another lobby. The normal visible countdown still runs:
as it runs down, additional AI fill seats left empty by humans. The initial
human target is a ceiling, not a requirement to wait indefinitely. This fallback
can start a smaller-human configuration even if that configuration already met
its quota; keeping games moving takes priority. Concurrent games can also exceed targets.

Once all five targets are met, new lobbies use normal countdown-based matchmaking. Set the
target to 0 to disable quotas. Other game sizes are unaffected.

Check current quotas from the command line using the same `.env` settings:

```bash
venv/bin/python scripts/show_quotas.py
```

To inspect a saved database instead, add `--db first100.db`. The report only
reads the database and does not start the server or change any records.
>>>>>>> outsider/main
