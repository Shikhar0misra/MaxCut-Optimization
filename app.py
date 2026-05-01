import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from graph_generator import generate_graph
from algorithms import brute_force_maxcut, greedy_maxcut
from database import init_db, insert_experiment, load_data
from analytics import runtime_prediction_plot, predict_runtime
from quantum_vqe import run_vqe


init_db()

st.title("Max-Cut Optimization Dashboard")

menu = st.sidebar.selectbox(
    "Menu",
    ["Generate Graphs", "Analytics"]
)

# ======================
# GENERATE GRAPH PAGE
# ======================

if menu == "Generate Graphs":

    nodes = st.slider("Nodes", 4, 8, 6)
    p = st.slider("Edge Probability", 0.1, 1.0, 0.5)
    num_graphs = st.number_input("Graphs",1,5,2)

    if st.button("Generate"):

        for i in range(num_graphs):

            G = generate_graph(nodes,p)

            brute_cut, brute_part, brute_time = brute_force_maxcut(G)
            greedy_cut, greedy_part, greedy_time = greedy_maxcut(G)

            vqe_energy, vqe_part = run_vqe(G)
            vqe_cut = -vqe_energy

            approx = greedy_cut / brute_cut if brute_cut else 0

            insert_experiment((
                nodes,
                len(G.edges()),
                brute_cut,
                greedy_cut,
                brute_time,
                greedy_time,
                approx,
                vqe_cut
            ))

            st.subheader(f"Graph {i+1}")

            pos = nx.spring_layout(G)

            # Greedy graph
            fig, ax = plt.subplots()
            colors = ["red" if greedy_part[i]==0 else "blue" for i in range(nodes)]
            nx.draw(G, pos, node_color=colors, with_labels=True, ax=ax)
            st.pyplot(fig)

            # VQE graph
            fig2, ax2 = plt.subplots()
            colors_vqe = ["green" if vqe_part[i]==0 else "yellow" for i in range(nodes)]
            nx.draw(G, pos, node_color=colors_vqe, with_labels=True, ax=ax2)
            st.pyplot(fig2)

            st.write("Optimal:", brute_cut)
            st.write("Greedy:", greedy_cut)
            st.write("VQE:", vqe_cut)

# ======================
# ANALYTICS PAGE
# ======================

if menu == "Analytics":

    df = load_data()

    if df.empty:
        st.warning("Generate graphs first")
        st.stop()

    st.dataframe(df)

    df_grouped = df.groupby("nodes").mean().reset_index()

    # Cuts
    fig1 = plt.figure()
    plt.bar(df_grouped["nodes"], df_grouped["brute_cut"], label="Optimal")
    plt.bar(df_grouped["nodes"], df_grouped["greedy_cut"], alpha=0.6)
    plt.bar(df_grouped["nodes"], df_grouped["vqe_cut"], alpha=0.6)
    plt.legend()
    st.pyplot(fig1)

    # Runtime
    fig2 = plt.figure()
    plt.plot(df_grouped["nodes"], df_grouped["brute_time"], label="Brute")
    plt.plot(df_grouped["nodes"], df_grouped["greedy_time"], label="Greedy")
    plt.legend()
    st.pyplot(fig2)

    # Prediction
    future, pred = predict_runtime(df_grouped)

    for n,p in zip(future,pred):
        st.write(f"Predicted time for {n} nodes:", p)

    fig3 = runtime_prediction_plot(df_grouped)
    st.pyplot(fig3)