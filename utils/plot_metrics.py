""" A script to plot validation metrics produced by the model.
"""
from collections import namedtuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_STEPS_PER_ITERATION = 5000

TraceSettings = namedtuple("TraceSettings", ["y", "name", "color"])


def plot_metrics(filename):
    df = pd.read_csv(filename, sep="\t")
    df["f1"] = 2 * (df.precision * df.recall) / (df.precision + df.recall)

    trace_metrics = [
        TraceSettings(y=df.recall, name="recall", color="red"),
        TraceSettings(y=df.precision, name="precision", color="blue"),
        TraceSettings(y=df.accuracy, name="accuracy", color="green"),
        TraceSettings(y=df.f1, name="f1", color="black"),
        TraceSettings(y=df.loss, name="loss", color="orange"),
        TraceSettings(y=df.roc_auc, name="roc_auc", color="darkcyan"),
    ]

    fig = go.Figure()

    for trace_metric in trace_metrics:
        fig.add_trace(
            go.Scatter(
                x=np.array(range(len(df))) * _STEPS_PER_ITERATION,
                y=trace_metric.y,
                mode="lines",
                name=trace_metric.name,
                line_color=trace_metric.color,
            )
        )

    fig.update_layout(title=filename, xaxis_title="iteration")

    return fig
