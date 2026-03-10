import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from graph_generator import generate_graph
from algorithms import brute_force_maxcut, greedy_maxcut
from database import init_db, insert_experiment, load_data
from analytics import runtime_prediction_plot, predict_runtime

init_db()

st.title("Max-Cut Optimization Dashboard")

menu = st.sidebar.selectbox(
    "Menu",
    ["Generate Graphs", "Analytics"]
)

# store graphs between reruns
if "graphs_data" not in st.session_state:
    st.session_state.graphs_data = []

# -------------------------
# GRAPH GENERATOR
# -------------------------

if menu == "Generate Graphs":

    st.header("Graph Generator")

    nodes = st.slider("Number of Nodes", 4, 12, 7)
    p = st.slider("Edge Probability", 0.1, 1.0, 0.5)
    num_graphs = st.number_input("Number of Graphs", 1, 20, 3)

    if st.button("Generate Graphs"):

        st.session_state.graphs_data = []

        for _ in range(num_graphs):

            G = generate_graph(nodes, p)

            brute_cut, brute_partition, brute_time = brute_force_maxcut(G)
            greedy_cut, greedy_partition, greedy_time = greedy_maxcut(G)

            approx = greedy_cut / brute_cut if brute_cut > 0 else 0

            insert_experiment((
                nodes,
                len(G.edges()),
                brute_cut,
                greedy_cut,
                brute_time,
                greedy_time,
                approx
            ))

            st.session_state.graphs_data.append({
                "G": G,
                "brute_cut": brute_cut,
                "greedy_cut": greedy_cut,
                "brute_partition": brute_partition,
                "greedy_partition": greedy_partition,
                "approx": approx
            })

    # -------------------------
    # DISPLAY GRAPHS
    # -------------------------

    for i, data in enumerate(st.session_state.graphs_data):

        G = data["G"]
        brute_cut = data["brute_cut"]
        greedy_cut = data["greedy_cut"]
        brute_partition = data["brute_partition"]
        greedy_partition = data["greedy_partition"]
        approx = data["approx"]

        st.subheader(f"Graph {i+1}")

        pos = nx.spring_layout(G)

        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, ax=ax)

        st.pyplot(fig)

        col1, col2 = st.columns(2)
        col1.metric("Optimal Cut", brute_cut)
        col2.metric("Greedy Cut", greedy_cut)

        st.write("Approx Ratio:", approx)

        # -------------------------
        # Optimal Cut Visualization
        # -------------------------

        with st.expander(f"View Optimal Cut Graph {i+1}"):

            fig_opt, ax_opt = plt.subplots()

            colors = [
                "red" if brute_partition[node] == 0 else "blue"
                for node in G.nodes()
            ]

            nx.draw(G, pos, node_color=colors, with_labels=True, ax=ax_opt)

            st.pyplot(fig_opt)

        # -------------------------
        # Greedy Cut Visualization
        # -------------------------

        with st.expander(f"View Greedy Cut Graph {i+1}"):

            fig_gr, ax_gr = plt.subplots()

            colors = [
                "green" if greedy_partition[node] == 0 else "orange"
                for node in G.nodes()
            ]

            nx.draw(G, pos, node_color=colors, with_labels=True, ax=ax_gr)

            st.pyplot(fig_gr)

        st.divider()

# -------------------------
# ANALYTICS PAGE
# -------------------------

if menu == "Analytics":

    st.header("Analytics Dashboard")

    df = load_data()

    st.dataframe(df)

    # Cut comparison
    fig1 = plt.figure()
    plt.bar(df["nodes"], df["brute_cut"], label="Optimal")
    plt.bar(df["nodes"], df["greedy_cut"], alpha=0.6, label="Greedy")
    plt.legend()
    plt.title("Cut Comparison")

    st.pyplot(fig1)

    # Runtime
    fig2 = plt.figure()
    plt.plot(df["nodes"], df["brute_time"], label="Brute")
    plt.plot(df["nodes"], df["greedy_time"], label="Greedy")
    plt.legend()
    plt.title("Runtime vs Nodes")

    st.pyplot(fig2)

    # Approx ratio
    fig3 = plt.figure()
    plt.plot(df["nodes"], df["approx_ratio"])
    plt.title("Approximation Ratio")

    st.pyplot(fig3)

    # Prediction
    future_nodes, prediction = predict_runtime(df)

    st.subheader("Runtime Prediction")

    for n, p in zip(future_nodes, prediction):
        st.write(f"Predicted brute runtime for {n} nodes:", p)

    st.subheader("Runtime Scaling Prediction")

    fig4 = runtime_prediction_plot(df)

    st.pyplot(fig4)