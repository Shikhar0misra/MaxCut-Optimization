import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def predict_runtime(df):

    df = df.sort_values(by="nodes")

    X = df["nodes"].values.reshape(-1,1)
    y = np.log(df["brute_time"].values + 1e-9)

    model = LinearRegression()
    model.fit(X, y)

    future_nodes = np.array([[15],[18],[20]])

    log_pred = model.predict(future_nodes)
    prediction = np.exp(log_pred)

    return future_nodes.flatten(), prediction


def runtime_prediction_plot(df):

    df = df.sort_values(by="nodes")

    nodes = df["nodes"].values.reshape(-1,1)

    brute = df["brute_time"].values
    greedy = df["greedy_time"].values

    brute_model = LinearRegression()
    brute_model.fit(nodes, np.log(brute + 1e-9))

    greedy_model = LinearRegression()
    greedy_model.fit(nodes, greedy)

    future_nodes = np.arange(min(nodes)[0], 25).reshape(-1,1)

    brute_pred = np.exp(brute_model.predict(future_nodes))
    greedy_pred = greedy_model.predict(future_nodes)

    plt.figure()

    plt.scatter(nodes, brute, color="red", label="Brute")
    plt.scatter(nodes, greedy, color="blue", label="Greedy")

    plt.plot(future_nodes, brute_pred, 'r--')
    plt.plot(future_nodes, greedy_pred, 'b--')

    plt.yscale("log")

    plt.title("Runtime Growth (Log Scale)")
    plt.xlabel("Nodes")
    plt.ylabel("Runtime")
    plt.legend()

    return plt