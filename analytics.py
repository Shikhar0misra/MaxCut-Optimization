import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def predict_runtime(df):

    X = df["nodes"].values.reshape(-1,1)
    y = df["brute_time"].values

    model = LinearRegression()
    model.fit(X, y)

    future_nodes = np.array([[15],[18],[20]])

    prediction = model.predict(future_nodes)

    return future_nodes.flatten(), prediction

def runtime_prediction_plot(df):

    nodes = df["nodes"].values.reshape(-1,1)

    brute = df["brute_time"].values
    greedy = df["greedy_time"].values

    # Train models
    brute_model = LinearRegression()
    greedy_model = LinearRegression()

    brute_model.fit(nodes, brute)
    greedy_model.fit(nodes, greedy)

    future_nodes = np.arange(min(nodes)[0], 20).reshape(-1,1)

    brute_pred = brute_model.predict(future_nodes)
    greedy_pred = greedy_model.predict(future_nodes)

    plt.figure()

    # actual points
    plt.scatter(nodes, brute, label="Brute Actual", color="red")
    plt.scatter(nodes, greedy, label="Greedy Actual", color="blue")

    # prediction curves
    plt.plot(future_nodes, brute_pred, color="red", linestyle="--", label="Brute Prediction")
    plt.plot(future_nodes, greedy_pred, color="blue", linestyle="--", label="Greedy Prediction")

    plt.title("Runtime Growth with Increasing Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Runtime (seconds)")
    plt.legend()

    return plt