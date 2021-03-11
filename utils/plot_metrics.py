""" A script to plot validation metrics produced by the model.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STEPS_PER_ITERATION = 5000

# NOTE: Replace these with your files.
files = [
    "new_sampler_0008.txt",
    "new_sampler_00008.txt",
    "old_sampler_0008.txt",
    "old_sampler_00008.txt",
]

# NOTE: Replace these with a grid you want to display.
rows = [1, 1, 2, 2]
cols = [1, 2, 1, 2]
fig = make_subplots(rows=2, cols=2, subplot_titles=files)

for index, filename in enumerate(files):
    df = pd.read_csv(filename, sep="\t")
    recall = df.recall
    precision = df.precision
    accuracy = df.accuracy
    f1 = 2 * (precision * recall) / (precision + recall)
    loss = df.loss

    fig.add_trace(
        go.Scatter(
            x=np.array(range(len(df))) * STEPS_PER_ITERATION,
            y=recall,
            mode="lines",
            name="recall",
            line_color="red",
            showlegend=(index == 0),
        ),
        row=rows[index],
        col=cols[index],
    )

    fig.add_trace(
        go.Scatter(
            x=np.array(range(len(df))) * STEPS_PER_ITERATION,
            y=precision,
            mode="lines",
            name="precision",
            line_color="blue",
            showlegend=(index == 0),
        ),
        row=rows[index],
        col=cols[index],
    )

    fig.add_trace(
        go.Scatter(
            x=np.array(range(len(df))) * STEPS_PER_ITERATION,
            y=accuracy,
            mode="lines",
            name="accuracy",
            line_color="green",
            showlegend=(index == 0),
        ),
        row=rows[index],
        col=cols[index],
    )

    fig.add_trace(
        go.Scatter(
            x=np.array(range(len(df))) * STEPS_PER_ITERATION,
            y=f1,
            mode="lines",
            name="F1",
            line_color="black",
            showlegend=(index == 0),
        ),
        row=rows[index],
        col=cols[index],
    )

    fig.add_trace(
        go.Scatter(
            x=np.array(range(len(df))) * STEPS_PER_ITERATION,
            y=loss,
            mode="lines",
            name="loss",
            line_color="orange",
            showlegend=(index == 0),
        ),
        row=rows[index],
        col=cols[index],
    )

    fig.update_xaxes(title_text="iteration", row=rows[index], col=cols[index])

fig.show()
