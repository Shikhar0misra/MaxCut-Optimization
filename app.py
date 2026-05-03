import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from graph_generator import generate_graph
from algorithms import brute_force_maxcut, greedy_maxcut
from database import init_db, insert_experiment, load_data
from analytics import (
    complexity_comparison,
    predict_all_runtimes,
    runtime_prediction_plot,
)
from quantum_qaoa import run_qaoa
from quantum_vqe import run_vqe


init_db()

st.title("Max-Cut Optimization Dashboard")

BRUTE_FORCE_NODE_LIMIT = 12
QUANTUM_SIM_NODE_LIMIT = 12

menu = st.sidebar.selectbox(
    "Menu",
    ["Generate Graphs", "Analytics"]
)

# ======================
# GENERATE GRAPH PAGE
# ======================

if menu == "Generate Graphs":

    nodes = st.slider("Nodes", 4, 50, 6)
    p = st.slider("Edge Probability", 0.1, 1.0, 0.5)
    num_graphs = st.number_input("Graphs",1,5,2)

    if nodes > BRUTE_FORCE_NODE_LIMIT:
        st.info(f"Brute-force optimal is skipped above {BRUTE_FORCE_NODE_LIMIT} nodes because it is O(2^n).")

    if nodes > QUANTUM_SIM_NODE_LIMIT:
        st.info(
            f"VQE/QAOA statevector simulation is skipped above {QUANTUM_SIM_NODE_LIMIT} nodes. "
            "A 50-qubit statevector cannot be simulated on a normal laptop."
        )

    if st.button("Generate"):

        for i in range(num_graphs):

            G = generate_graph(nodes,p)

            brute_cut = None
            brute_part = None
            brute_time = None
            if nodes <= BRUTE_FORCE_NODE_LIMIT:
                brute_cut, brute_part, brute_time = brute_force_maxcut(G)

            greedy_cut, greedy_part, greedy_time = greedy_maxcut(G)

            vqe_part = None
            vqe_time = None
            vqe_quantum_time = None
            raw_vqe_cut = None
            vqe_cut = None
            qaoa_part = None
            qaoa_time = None
            qaoa_quantum_time = None
            raw_qaoa_cut = None
            qaoa_cut = None

            if nodes <= QUANTUM_SIM_NODE_LIMIT:
                vqe_energy, vqe_part, vqe_time, vqe_quantum_time, raw_vqe_cut, vqe_cut = run_vqe(G)
                qaoa_cut, qaoa_part, qaoa_time, qaoa_quantum_time, raw_qaoa_cut = run_qaoa(G)

            approx = greedy_cut / brute_cut if brute_cut else None

            insert_experiment((
                nodes,
                len(G.edges()),
                brute_cut,
                greedy_cut,
                brute_time,
                greedy_time,
                vqe_time,
                qaoa_time,
                vqe_quantum_time,
                qaoa_quantum_time,
                approx,
                vqe_cut,
                qaoa_cut
            ))

            st.subheader(f"Graph {i+1}")

            pos = nx.spring_layout(G)

            # Greedy graph
            fig, ax = plt.subplots()
            colors = ["red" if greedy_part[i]==0 else "blue" for i in range(nodes)]
            nx.draw(G, pos, node_color=colors, with_labels=True, ax=ax)
            st.pyplot(fig)

            if vqe_part is not None:
                # VQE graph
                fig2, ax2 = plt.subplots()
                colors_vqe = ["green" if vqe_part[i]==0 else "yellow" for i in range(nodes)]
                nx.draw(G, pos, node_color=colors_vqe, with_labels=True, ax=ax2)
                st.pyplot(fig2)

            if qaoa_part is not None:
                # QAOA graph
                fig3, ax3 = plt.subplots()
                colors_qaoa = ["purple" if qaoa_part[i]==0 else "orange" for i in range(nodes)]
                nx.draw(G, pos, node_color=colors_qaoa, with_labels=True, ax=ax3)
                st.pyplot(fig3)

            st.write("Optimal:", brute_cut if brute_cut is not None else "Skipped")
            st.write("Greedy:", greedy_cut)
            st.write("Raw VQE:", raw_vqe_cut if raw_vqe_cut is not None else "Skipped")
            st.write("VQE + local improvement:", vqe_cut if vqe_cut is not None else "Skipped")
            st.write("Raw QAOA:", raw_qaoa_cut if raw_qaoa_cut is not None else "Skipped")
            st.write("QAOA + local improvement:", qaoa_cut if qaoa_cut is not None else "Skipped")
            st.write("Brute time:", brute_time if brute_time is not None else "Skipped")
            st.write("Greedy time:", greedy_time)
            st.write("VQE time:", vqe_quantum_time if vqe_quantum_time is not None else "Skipped")
            st.write("QAOA time:", qaoa_quantum_time if qaoa_quantum_time is not None else "Skipped")

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
    df_grouped["vqe_approx_ratio"] = (
        df_grouped["vqe_cut"] / df_grouped["brute_cut"]
    ).where(df_grouped["brute_cut"] != 0)
    if "qaoa_cut" in df_grouped.columns:
        df_grouped["qaoa_approx_ratio"] = (
            df_grouped["qaoa_cut"] / df_grouped["brute_cut"]
        ).where(df_grouped["brute_cut"] != 0)

    # Cuts
    fig1 = plt.figure()
    plt.bar(df_grouped["nodes"], df_grouped["brute_cut"], label="Optimal")
    plt.bar(df_grouped["nodes"], df_grouped["greedy_cut"], alpha=0.6, label="Greedy")
    plt.bar(df_grouped["nodes"], df_grouped["vqe_cut"], alpha=0.6, label="VQE + Local")
    if "qaoa_cut" in df_grouped.columns and df_grouped["qaoa_cut"].notna().any():
        plt.bar(df_grouped["nodes"], df_grouped["qaoa_cut"], alpha=0.6, label="QAOA + Local")
    plt.xlabel("Nodes")
    plt.ylabel("Cut Value")
    plt.legend()
    st.pyplot(fig1)

    # Approximation ratios
    st.subheader("Approximation Ratio")
    fig_ratio = plt.figure()
    plt.plot(df_grouped["nodes"], df_grouped["approx_ratio"], marker="o", label="Greedy")
    plt.plot(df_grouped["nodes"], df_grouped["vqe_approx_ratio"], marker="o", label="VQE + Local")
    if "qaoa_approx_ratio" in df_grouped.columns and df_grouped["qaoa_approx_ratio"].notna().any():
        plt.plot(df_grouped["nodes"], df_grouped["qaoa_approx_ratio"], marker="o", label="QAOA + Local")
    plt.xlabel("Nodes")
    plt.ylabel("Approximation Ratio")
    plt.legend()
    st.pyplot(fig_ratio)

    # Runtime
    st.subheader("Runtime")
    st.caption("VQE and QAOA times estimate quantum circuit execution only; simulator time is not included.")
    fig2 = plt.figure()
    plt.plot(df_grouped["nodes"], df_grouped["brute_time"], label="Brute")
    plt.plot(df_grouped["nodes"], df_grouped["greedy_time"], label="Greedy")
    if "vqe_quantum_time" in df_grouped.columns and df_grouped["vqe_quantum_time"].notna().any():
        plt.plot(df_grouped["nodes"], df_grouped["vqe_quantum_time"], label="VQE")
    if "qaoa_quantum_time" in df_grouped.columns and df_grouped["qaoa_quantum_time"].notna().any():
        plt.plot(df_grouped["nodes"], df_grouped["qaoa_quantum_time"], label="QAOA")
    plt.xlabel("Nodes")
    plt.ylabel("Runtime")
    plt.legend()
    st.pyplot(fig2)

    # Prediction
    predictions = predict_all_runtimes(df_grouped)

    for algorithm, (future, pred) in predictions.items():
        st.subheader(f"{algorithm} Runtime Prediction")
        for n,p in zip(future,pred):
            st.write(f"Predicted time for {n} nodes:", p)

    fig3 = runtime_prediction_plot(df_grouped)
    st.pyplot(fig3)

    st.subheader("Time Complexity")
    st.table(complexity_comparison())
