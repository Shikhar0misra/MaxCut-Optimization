import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx

from graph_generator import generate_graph
from algorithms import brute_force_maxcut, greedy_maxcut
from database import init_db, insert_experiment, load_data
from analytics import (
    complexity_comparison,
    predict_all_runtimes,
    runtime_prediction_plot,
)
from quantum_vqe import run_vqe


st.set_page_config(
    page_title="Max-Cut Optimization",
    page_icon="M",
    layout="wide",
)

init_db()

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(45, 212, 191, 0.28), transparent 34rem),
                radial-gradient(circle at top right, rgba(96, 165, 250, 0.24), transparent 32rem),
                linear-gradient(135deg, #f8fafc 0%, #eef6ff 45%, #f7f3ff 100%);
        }

        h1, h2, h3 {
            color: #1f2937;
            letter-spacing: 0;
        }

        [data-testid="stSidebar"] {
            background: #111827;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: #f9fafb;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #e5e7eb;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Max-Cut Optimization Dashboard")
st.caption("Compare classical and variational quantum approaches on generated Max-Cut graphs.")

BRUTE_FORCE_NODE_LIMIT = 16
QUANTUM_SIM_NODE_LIMIT = 14

st.sidebar.title("Controls")
menu = st.sidebar.selectbox(
    "Menu",
    ["Generate Graphs", "Analytics"]
)


def display_value(value):
    if value is None:
        return "Skipped"
    if isinstance(value, float):
        return f"{value:.6g}"
    return value

# ======================
# GENERATE GRAPH PAGE
# ======================

if menu == "Generate Graphs":

    st.subheader("Generate Graphs")
    control_panel = st.container(border=True)
    with control_panel:
        input_cols = st.columns([2, 2, 1])
        with input_cols[0]:
            nodes = st.slider("Nodes", 4, 50, 6)
        with input_cols[1]:
            p = st.slider("Edge Probability", 0.1, 1.0, 0.5)
        with input_cols[2]:
            num_graphs = st.number_input("Graphs", 1, 5, 2)

    if nodes > BRUTE_FORCE_NODE_LIMIT:
        st.info(f"Brute-force optimal is skipped above {BRUTE_FORCE_NODE_LIMIT} nodes because it is O(2^n).")

    if nodes > QUANTUM_SIM_NODE_LIMIT:
        st.info(
            f"VQE statevector simulation is skipped above {QUANTUM_SIM_NODE_LIMIT} nodes. "
            "A 50-qubit statevector cannot be simulated on a normal laptop."
        )

    if st.button("Generate", type="primary"):

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

            if nodes <= QUANTUM_SIM_NODE_LIMIT:
                vqe_energy, vqe_part, vqe_time, vqe_quantum_time, raw_vqe_cut, vqe_cut = run_vqe(G)

            approx = greedy_cut / brute_cut if brute_cut else None

            insert_experiment((
                nodes,
                len(G.edges()),
                brute_cut,
                greedy_cut,
                brute_time,
                greedy_time,
                vqe_time,
                vqe_quantum_time,
                approx,
                vqe_cut
            ))

            st.divider()
            st.subheader(f"Graph {i+1}")

            metric_cols = st.columns(6)
            metric_cols[0].metric("Nodes", nodes)
            metric_cols[1].metric("Edges", len(G.edges()))
            metric_cols[2].metric("Optimal Cut", display_value(brute_cut))
            metric_cols[3].metric("Greedy Cut", display_value(greedy_cut))
            metric_cols[4].metric("Raw VQE Cut", display_value(raw_vqe_cut))
            metric_cols[5].metric("VQE + Local", display_value(vqe_cut))

            pos = nx.spring_layout(G)

            # Greedy graph
            fig, ax = plt.subplots()
            fig.patch.set_facecolor("#ffffff")
            colors = ["#ef4444" if greedy_part[i]==0 else "#2563eb" for i in range(nodes)]
            nx.draw(G, pos, node_color=colors, with_labels=True, ax=ax)
            ax.set_title("Greedy Max-Cut Partition")
            ax.legend(
                handles=[
                    Patch(color="#ef4444", label="Set 0"),
                    Patch(color="#2563eb", label="Set 1"),
                ],
                loc="upper right",
            )

            if vqe_part is not None:
                # VQE graph
                fig2, ax2 = plt.subplots()
                fig2.patch.set_facecolor("#ffffff")
                colors_vqe = ["#16a34a" if vqe_part[i]==0 else "#f59e0b" for i in range(nodes)]
                nx.draw(G, pos, node_color=colors_vqe, with_labels=True, ax=ax2)
                ax2.set_title("VQE + Local Improvement Partition")
                ax2.legend(
                    handles=[
                        Patch(color="#16a34a", label="Set 0"),
                        Patch(color="#f59e0b", label="Set 1"),
                    ],
                    loc="upper right",
                )
                graph_cols = st.columns(2)
                graph_cols[0].pyplot(fig, use_container_width=True)
                graph_cols[1].pyplot(fig2, use_container_width=True)
            else:
                st.pyplot(fig, use_container_width=True)

            time_cols = st.columns(3)
            time_cols[0].metric("Brute Time", display_value(brute_time))
            time_cols[1].metric("Greedy Time", display_value(greedy_time))
            time_cols[2].metric("VQE Time", display_value(vqe_quantum_time))

# ======================
# ANALYTICS PAGE
# ======================

if menu == "Analytics":

    df = load_data()

    if df.empty:
        st.warning("Generate graphs first")
        st.stop()

    st.subheader("Analytics")
    st.dataframe(df, use_container_width=True)

    df_grouped = df.groupby("nodes").mean().reset_index()
    df_grouped["vqe_approx_ratio"] = (
        df_grouped["vqe_cut"] / df_grouped["brute_cut"]
    ).where(df_grouped["brute_cut"] != 0)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Experiments", len(df))
    summary_cols[1].metric("Node Counts", df["nodes"].nunique())
    summary_cols[2].metric("Max Nodes", int(df["nodes"].max()))
    summary_cols[3].metric("Avg Edges", f"{df['edges'].mean():.2f}")

    # Cuts
    fig1 = plt.figure(facecolor="#ffffff")
    plt.bar(df_grouped["nodes"], df_grouped["brute_cut"], label="Optimal")
    plt.bar(df_grouped["nodes"], df_grouped["greedy_cut"], alpha=0.6, label="Greedy")
    plt.bar(df_grouped["nodes"], df_grouped["vqe_cut"], alpha=0.6, label="VQE + Local")
    plt.xlabel("Nodes")
    plt.ylabel("Cut Value")
    plt.title("Cut Quality by Node Count")
    plt.legend()

    # Approximation ratios
    fig_ratio = plt.figure(facecolor="#ffffff")
    plt.plot(df_grouped["nodes"], df_grouped["approx_ratio"], marker="o", label="Greedy")
    plt.plot(df_grouped["nodes"], df_grouped["vqe_approx_ratio"], marker="o", label="VQE + Local")
    plt.xlabel("Nodes")
    plt.ylabel("Approximation Ratio")
    plt.title("Approximation Ratio")
    plt.legend()

    # Runtime
    fig2 = plt.figure(facecolor="#ffffff")
    plt.plot(df_grouped["nodes"], df_grouped["brute_time"], label="Brute")
    plt.plot(df_grouped["nodes"], df_grouped["greedy_time"], label="Greedy")
    if "vqe_quantum_time" in df_grouped.columns and df_grouped["vqe_quantum_time"].notna().any():
        plt.plot(df_grouped["nodes"], df_grouped["vqe_quantum_time"], label="VQE")
    plt.xlabel("Nodes")
    plt.ylabel("Runtime")
    plt.title("Runtime by Node Count")
    plt.legend()

    chart_tabs = st.tabs(["Cut Quality", "Approximation Ratio", "Runtime"])
    with chart_tabs[0]:
        st.pyplot(fig1, use_container_width=True)
    with chart_tabs[1]:
        st.pyplot(fig_ratio, use_container_width=True)
    with chart_tabs[2]:
        st.caption("VQE time estimates quantum circuit execution only; simulator time is not included.")
        st.pyplot(fig2, use_container_width=True)

    # Prediction
    st.subheader("Runtime Prediction")
    prediction_cols = st.columns([1, 2])
    with prediction_cols[0]:
        prediction_node = st.selectbox(
            "Nodes to predict",
            list(range(4, 101)),
            index=96,
        )
    predictions = predict_all_runtimes(df_grouped, [prediction_node])

    prediction_rows = []
    for algorithm, (future, pred) in predictions.items():
        prediction_rows.append({
            "Algorithm": algorithm,
            "Nodes": int(future[0]),
            "Predicted Time": float(pred[0]),
        })

    with prediction_cols[1]:
        st.table(prediction_rows)

    fig3 = runtime_prediction_plot(df_grouped)
    st.pyplot(fig3, use_container_width=True)

    st.subheader("Time Complexity")
    st.table(complexity_comparison())
