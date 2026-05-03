import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


RUNTIME_COLUMNS = {
    "brute_time": "Brute Force",
    "greedy_time": "Greedy",
    "vqe_quantum_time": "VQE",
    "qaoa_quantum_time": "QAOA",
}


def predict_runtime(df, runtime_column="brute_time"):

    df = df.sort_values(by="nodes")
    df = df.dropna(subset=[runtime_column])

    X = df["nodes"].values.reshape(-1,1)
    y = np.log(df[runtime_column].values + 1e-9)

    model = LinearRegression()
    model.fit(X, y)

    future_nodes = np.array([[15],[18],[20]])

    log_pred = model.predict(future_nodes)
    prediction = np.exp(log_pred)

    return future_nodes.flatten(), prediction


def predict_all_runtimes(df):

    predictions = {}

    for column, label in RUNTIME_COLUMNS.items():
        if column in df.columns and df[column].notna().any():
            future, pred = predict_runtime(df, column)
            predictions[label] = (future, pred)

    return predictions


def complexity_comparison():

    return [
        {
            "Algorithm": "Brute Force",
            "Time Complexity": "O(2^n * m)",
            "Why": "Checks every partition and evaluates all m edges for each one.",
            "Space Complexity": "O(n)",
        },
        {
            "Algorithm": "Greedy",
            "Time Complexity": "O(k * n * m)",
            "Why": "For k improvement rounds, each node flip recomputes the cut over all m edges.",
            "Space Complexity": "O(n)",
        },
        {
            "Algorithm": "VQE + Local Improvement",
            "Time Complexity": "O(I * 2^n * p(n + m) + k * n * m)",
            "Why": "Statevector VQE runs up to I optimizer iterations, then a cheap local search polishes the measured bitstring.",
            "Space Complexity": "O(2^n) in this statevector simulator",
        },
        {
            "Algorithm": "QAOA + Local Improvement",
            "Time Complexity": "O(R * I * 2^n * p(n + m) + k * n * m)",
            "Why": "Runs R QAOA restarts; each optimizer step simulates p cost/mixer layers, then local search polishes the best bitstring.",
            "Space Complexity": "O(2^n) in this statevector simulator",
        },
    ]


def runtime_prediction_plot(df):

    df = df.sort_values(by="nodes")

    nodes = df["nodes"].values.reshape(-1,1)

    future_nodes = np.arange(min(nodes)[0], 25).reshape(-1,1)

    fig, ax = plt.subplots()

    colors = {
        "brute_time": "red",
        "greedy_time": "blue",
        "vqe_quantum_time": "green",
        "qaoa_quantum_time": "purple",
    }

    for column, label in RUNTIME_COLUMNS.items():
        if column not in df.columns:
            continue

        series = df[column].dropna()
        if series.empty:
            continue

        valid = df.dropna(subset=[column])
        valid_nodes = valid["nodes"].values.reshape(-1,1)
        runtime = valid[column].values

        model = LinearRegression()
        model.fit(valid_nodes, np.log(runtime + 1e-9))
        prediction = np.exp(model.predict(future_nodes))

        ax.scatter(valid_nodes, runtime, color=colors[column], label=label)
        ax.plot(future_nodes, prediction, linestyle="--", color=colors[column])

    ax.set_yscale("log")

    ax.set_title("Runtime Growth (Log Scale)")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Runtime")
    ax.legend()

    return fig
