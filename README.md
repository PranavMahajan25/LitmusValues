# LitmusValues: An Evaluation Pipeline to Reveal AI Value Preference

![Concept image](img/concept.png)

## Pre-requisites

1. Please do ``` pip install -r requirements.txt``` ideally in a conda/venv environment

2. Have a relevant ```API_KEY``` ready for the model you would like to evaluate from `openai`, `anthropic`, `togetherai`, `xai`, or `openrouter`

## Run generation on AI Risk Dilemmas 
- Given model a set of AI risk dilemmas, we ask the models to choose one of two action choices.

### Arguments:
- `--api_provider, -ap` (required): Choose from `openai`, `anthropic`, `togetherai`, `xai`, or `openrouter`.
- `--api_key, -ak` (required): API key for the selected provider.
- `--model, -m` (required): Name of the model to use.
- `--generations_dir, -g` (optional): Directory to save output generations. Default is `generations`.
- `--num_parallel_request, -n` (optional): Number of parallel requests to make. Default is `1`.
- `--debug, -d` (optional): Run in debug mode with only 5 examples.

### Example:
```bash
python run_ai_risk_dilemmas.py --api_provider openai --model gpt-4o --api_key sk-...
```

## Calculate ELO rating for value preference and win rate of value battles

- Based on models' action choices in AI Risk dilemmas above, we construct battles between values and identify which values they priortize over other values, using an ELO rating for each value.

### Arguments:
- `--model, -m` (required): Name of the model to evaluate.
- `--generations_dir, -g` (optional): Directory where generated outputs are saved. Default is `generations`.
- `--elo_rating_dir, -e` (optional): Directory to save ELO rating results. Default is `elo_rating`.

### Example:
```bash
python calculate_elo_rating.py --model gpt-4o 
```

## Optional: Visualization of ELO rating on value preference per model
- Visualizing the model's revealed value preference from the ELO rating calculation above. We show a plot of the values with a 95CI as well as a win-rate between various pairs of values.

### Arguments:
- `--model, -m` (required): Name of the model to evaluate.
- `--generations_dir, -g` (optional): Directory containing generated outputs. Default is `generations`.
- `--output_elo_fig_dir, -f` (optional): Directory to save ELO score figures. Default is `output_elo_figs`.
- `--output_win_rate_fig_dir, -w` (optional): Directory to save win-rate figures. Default is `output_win_rate_figs`.

### Example:
```bash
python visualize_elo_rating.py --model gpt-4o 
```