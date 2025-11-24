#!/usr/bin/env python3
# visualize_elo_rating_stated.py
"""
Visualize Elo ratings and predicted win-rate matrix for stated-preferences outputs.

Reads Elo estimates from an Elo CSV (produced by calculate_elo_rating_stated.py)
and draws:
 - a ranked Elo bar/annotation plot (PNG)
 - a predicted win-rate heatmap (PNG)

Usage:
  python visualize_elo_rating_stated.py --model gpt-4o \
      --elo_rating_dir elo_rating_stated \
      --output_elo_fig_dir output_elo_figs_stated \
      --output_win_rate_fig_dir output_winrate_figs_stated
"""
import argparse
import os

import numpy as np
import pandas as pd
import plotly.express as px

parser = argparse.ArgumentParser(description='visualize elo ratings and win rates (stated prefs)')
parser.add_argument("--model", "-m", required=True)
parser.add_argument("--elo_rating_dir", "-e", default="elo_rating_stated")
parser.add_argument("--output_elo_fig_dir","-f", default="output_elo_figs_stated")
parser.add_argument("--output_win_rate_fig_dir","-w", default="output_win_rate_figs_stated")
parser.add_argument("--max_values_to_show", type=int, default=30, help="Max number of value classes to include in the win-rate heatmap")
args = parser.parse_args()

model : str = args.model
model = model.replace('/', '__')
elo_rating_dir = args.elo_rating_dir
output_elo_fig_dir = args.output_elo_fig_dir
output_win_rate_fig_dir = args.output_win_rate_fig_dir
max_values_to_show = args.max_values_to_show

os.makedirs(output_elo_fig_dir, exist_ok=True)
os.makedirs(output_win_rate_fig_dir, exist_ok=True)

input_elo_csv = os.path.join(elo_rating_dir, f"{model}.csv")
output_elo_fig_path = os.path.join(output_elo_fig_dir, f"{model}.png")
output_win_rate_fig_path = os.path.join(output_win_rate_fig_dir, f"{model}.png")

if not os.path.exists(input_elo_csv):
    raise FileNotFoundError(f"Elo CSV not found: {input_elo_csv}")

# Load Elo CSV: expected columns include 'value_class' and 'Elo Rating'
elo_df = pd.read_csv(input_elo_csv)
if 'value_class' not in elo_df.columns or 'Elo Rating' not in elo_df.columns:
    raise ValueError(f"Unexpected Elo CSV format. Columns: {elo_df.columns.tolist()}")

# Ensure sorting by Elo desc (rank 1 = highest Elo)
elo_df_sorted = elo_df.sort_values(by='Elo Rating', ascending=False).reset_index(drop=True)
# Prepare bar dataframe
bars = pd.DataFrame({
    'value_class': elo_df_sorted['value_class'].astype(str),
    'rating': elo_df_sorted['Elo Rating'].astype(float)
})
bars['rating_rounded'] = np.round(bars['rating'], 2)

# Bar chart (annotated)
fig = px.scatter(bars, x='value_class', y='rating', title=f"{model}: Elo Ratings (stated prefs)")
# add annotations for numeric labels
for i, row in bars.iterrows():
    fig.add_annotation(
        x=row['value_class'],
        y=row['rating'],
        text=str(row['rating_rounded']),
        showarrow=False,
        textangle=-90,
        font=dict(size=12, color="black"),
        yshift=0
    )

fig.update_layout(
    xaxis_title="Value Class",
    yaxis_title="Elo Rating",
    height=480,
    width=900,
)
fig.update_xaxes(tickangle=-90)
fig.write_image(output_elo_fig_path)
print(f"Wrote Elo figure to {output_elo_fig_path}")

# -----------------------------
# Compute predicted win-rate matrix from Elo ratings
# -----------------------------
# Build elo dict
elo_map = dict(zip(bars['value_class'], bars['rating']))

def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = {a: {} for a in names}
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
    # build DataFrame
    df = pd.DataFrame(wins).T  # value_1 as rows, value_2 as columns
    df.index.name = 'value_1'
    df.columns.name = 'value_2'
    return df

win_rate = predict_win_rate(elo_map)

# Limit to top N names by average win rate (to keep heatmap readable)
ordered_models = win_rate.mean(axis=1).sort_values(ascending=False).index
ordered_models = list(ordered_models[:max_values_to_show])

win_rate_sub = win_rate.loc[ordered_models, ordered_models]

fig2 = px.imshow(
    win_rate_sub,
    color_continuous_scale='RdBu',
    text_auto=".2f",
    title=f"{model}: Predicted Win Rate Using Elo Ratings",
    labels={'x':'Value 2', 'y':'Value 1', 'color':'Win Rate'}
)
fig2.update_layout(xaxis_side="top", height=900, width=900)
fig2.update_traces(hovertemplate="Value 1: %{y}<br>Value 2: %{x}<br>Win Rate: %{z:.2f}<extra></extra>")
fig2.write_image(output_win_rate_fig_path)
print(f"Wrote Win-rate heatmap to {output_win_rate_fig_path}")
