import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np

from graph_generator import generate_graph
from algorithms import brute_force_maxcut, greedy_maxcut, format_duration, BRUTE_FORCE_LIMIT
from database import init_db, insert_experiment, load_data
from analytics import runtime_prediction_plot, predict_runtime

st.set_page_config(
    page_title="MaxCut Lab · Research Interface",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;1,9..144,300&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --ink:        #0a0c10;
    --paper:      #f5f2eb;
    --cream:      #ede9df;
    --warm-mid:   #c8bfa8;
    --muted:      #8a8070;
    --accent:     #1a3a5c;
    --cut:        #c0392b;
    --greedy:     #1a6b4a;
    --gold:       #b8860b;
    --border:     #d4cfc4;
    --mono:       'IBM Plex Mono', monospace;
    --serif:      'Fraunces', Georgia, serif;
    --sans:       'DM Sans', sans-serif;
    --shadow-sm:  0 1px 4px rgba(10,12,16,0.08);
}

html, body, [class*="css"] { font-family:var(--sans) !important; background-color:var(--paper) !important; color:var(--ink) !important; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding:2.5rem 3rem 5rem !important; max-width:1340px; }

[data-testid="stSidebar"] { background:var(--ink) !important; border-right:none !important; }
[data-testid="stSidebar"] * { color:var(--paper) !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div { background:#1a1f2b !important; border:1px solid #2e3545 !important; }

.sidebar-brand { display:flex; flex-direction:column; padding:1.5rem 0 1.8rem; border-bottom:1px solid #2e3545; margin-bottom:1.8rem; }
.sidebar-brand-title { font-family:var(--mono); font-size:1rem; font-weight:700; letter-spacing:0.3em; text-transform:uppercase; }
.sidebar-brand-subtitle { font-family:var(--mono); font-size:0.6rem; letter-spacing:0.18em; color:#5a6480 !important; text-transform:uppercase; margin-top:0.2rem; }
.sidebar-badge { display:inline-block; background:var(--cut); color:#fff !important; font-family:var(--mono); font-size:0.55rem; letter-spacing:0.12em; padding:0.2rem 0.55rem; border-radius:2px; margin-top:0.6rem; width:fit-content; }
.sidebar-nav-label { font-family:var(--mono); font-size:0.6rem; letter-spacing:0.18em; color:#5a6480 !important; text-transform:uppercase; margin-bottom:0.4rem; }

.hero { background:var(--ink); border-radius:12px; padding:2.8rem 3rem 2.4rem; margin-bottom:2.5rem; position:relative; overflow:hidden; }
.hero::before { content:''; position:absolute; top:-40px; right:-40px; width:280px; height:280px; border-radius:50%; border:1px solid #2e3545; }
.hero::after  { content:''; position:absolute; bottom:-60px; right:80px; width:180px; height:180px; border-radius:50%; border:1px solid #2e3545; }
.hero-eyebrow { font-family:var(--mono); font-size:0.65rem; letter-spacing:0.25em; color:var(--cut); text-transform:uppercase; margin-bottom:0.8rem; }
.hero-title   { font-family:var(--serif); font-size:2.6rem; font-weight:700; color:var(--paper); line-height:1.1; margin-bottom:0.5rem; }
.hero-title em{ font-style:italic; color:#7aabce; }
.hero-desc    { font-family:var(--sans); font-size:0.9rem; color:#8a95aa; max-width:520px; line-height:1.7; }
.hero-rule    { width:48px; height:2px; background:var(--cut); margin:1.2rem 0; }

.sec-head { display:flex; align-items:center; gap:0.8rem; margin:2.2rem 0 1rem; }
.sec-head-num  { font-family:var(--mono); font-size:0.6rem; font-weight:600; color:#fff; background:var(--accent); padding:0.25rem 0.5rem; border-radius:3px; letter-spacing:0.05em; }
.sec-head-text { font-family:var(--mono); font-size:0.68rem; font-weight:600; letter-spacing:0.2em; color:var(--muted); text-transform:uppercase; }

.control-panel { background:var(--cream); border:1px solid var(--border); border-top:3px solid var(--accent); border-radius:8px; padding:1.6rem 1.8rem 1.2rem; margin-bottom:1.8rem; box-shadow:var(--shadow-sm); }
.control-panel-title { font-family:var(--mono); font-size:0.6rem; letter-spacing:0.22em; color:var(--muted); text-transform:uppercase; margin-bottom:1rem; }

.complexity-pill { display:inline-block; font-family:var(--mono); font-size:0.6rem; letter-spacing:0.1em; padding:0.2rem 0.65rem; border-radius:20px; margin-top:0.3rem; font-weight:600; }
.complexity-feasible { background:#e6f4ed; color:#1a6b4a; border:1px solid #a8d5bb; }
.complexity-warn     { background:#fff3e0; color:#b8860b; border:1px solid #f0c070; }
.complexity-danger   { background:#fdecea; color:#c0392b; border:1px solid #f0a0a0; }

[data-testid="metric-container"] { background:var(--cream) !important; border:1px solid var(--border) !important; border-radius:6px !important; padding:1rem 1.2rem !important; box-shadow:var(--shadow-sm) !important; }
[data-testid="metric-container"] [data-testid="stMetricLabel"] { font-family:var(--mono) !important; font-size:0.58rem !important; letter-spacing:0.15em !important; color:var(--muted) !important; text-transform:uppercase !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family:var(--mono) !important; font-size:1.8rem !important; font-weight:700 !important; color:var(--ink) !important; }

.graph-card { background:#fff; border:1px solid var(--border); border-radius:10px; padding:1.6rem 1.8rem 1.2rem; margin-bottom:1.5rem; box-shadow:var(--shadow-sm); position:relative; }
.graph-card-tag { font-family:var(--mono); font-size:0.58rem; letter-spacing:0.2em; color:var(--muted); text-transform:uppercase; }
.graph-card-num { position:absolute; top:1.2rem; right:1.6rem; font-family:var(--serif); font-style:italic; font-size:3rem; font-weight:300; color:var(--border); line-height:1; }

.bf-warning { background:#fff8f0; border:1px solid #e8a44a; border-left:4px solid #e8a44a; border-radius:8px; padding:1.2rem 1.5rem; margin-bottom:1rem; }
.bf-warning-title { font-family:var(--mono); font-size:0.65rem; font-weight:700; letter-spacing:0.15em; color:#a0600a; text-transform:uppercase; margin-bottom:0.5rem; }
.bf-warning-body  { font-family:var(--sans); font-size:0.85rem; color:#7a4a0a; line-height:1.6; }
.bf-time-badge { display:inline-block; background:#a0600a; color:#fff; font-family:var(--mono); font-size:0.75rem; font-weight:700; padding:0.3rem 0.9rem; border-radius:4px; margin-top:0.7rem; }

.ratio-card  { background:var(--ink); border-radius:8px; padding:1rem 1.2rem; }
.ratio-label { font-family:var(--mono); font-size:0.55rem; letter-spacing:0.18em; color:#5a6480; text-transform:uppercase; margin-bottom:0.4rem; }
.ratio-value { font-family:var(--mono); font-size:1.9rem; font-weight:700; color:#7aabce; line-height:1; }
.ratio-bar-wrap { margin-top:0.6rem; background:#1a1f2b; border-radius:2px; height:3px; }
.ratio-bar-fill { height:3px; border-radius:2px; }

.legend-row { display:flex; gap:1rem; margin-top:0.5rem; }
.legend-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }
.legend-txt { font-family:var(--mono); font-size:0.65rem; color:var(--muted); }

[data-testid="stExpander"] { background:var(--cream) !important; border:1px solid var(--border) !important; border-radius:8px !important; }
[data-testid="stExpander"] summary { font-family:var(--mono) !important; font-size:0.7rem !important; letter-spacing:0.1em !important; color:var(--muted) !important; }

[data-testid="stSlider"] label, [data-testid="stNumberInput"] label { font-family:var(--mono) !important; font-size:0.7rem !important; letter-spacing:0.1em !important; color:var(--muted) !important; text-transform:uppercase !important; }

[data-testid="stButton"] > button { background:var(--ink) !important; color:var(--paper) !important; font-family:var(--mono) !important; font-weight:600 !important; font-size:0.72rem !important; letter-spacing:0.15em !important; text-transform:uppercase !important; border:none !important; border-radius:5px !important; padding:0.6rem 1.8rem !important; }
[data-testid="stButton"] > button:hover { background:var(--accent) !important; }

hr { border-color:var(--border) !important; }
[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:8px !important; overflow:hidden; box-shadow:var(--shadow-sm); }

.insight-card { background:var(--ink); border-radius:8px; padding:1.2rem 1.5rem; margin:1rem 0 1.5rem; }
.insight-card-label { font-family:var(--mono); font-size:0.55rem; letter-spacing:0.22em; color:var(--cut); text-transform:uppercase; margin-bottom:0.5rem; }
.insight-card-body  { font-family:var(--sans); font-size:0.85rem; color:#aab4c8; line-height:1.7; }

.stat-row  { display:flex; gap:1px; margin-bottom:2rem; background:var(--border); border-radius:8px; overflow:hidden; border:1px solid var(--border); }
.stat-cell { flex:1; background:#fff; padding:1rem 1.2rem; text-align:center; }
.stat-cell:first-child { border-radius:7px 0 0 7px; }
.stat-cell:last-child  { border-radius:0 7px 7px 0; }
.stat-num  { font-family:var(--mono); font-size:1.5rem; font-weight:700; color:var(--ink); }
.stat-lbl  { font-family:var(--mono); font-size:0.55rem; letter-spacing:0.15em; color:var(--muted); text-transform:uppercase; margin-top:0.3rem; }

.footnote { font-family:var(--mono); font-size:0.6rem; color:var(--warm-mid); letter-spacing:0.08em; margin-top:3rem; padding-top:1rem; border-top:1px solid var(--border); }
</style>
""", unsafe_allow_html=True)

mpl.rcParams.update({
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#d4cfc4", "axes.labelcolor": "#8a8070",
    "xtick.color": "#8a8070", "ytick.color": "#8a8070",
    "text.color": "#0a0c10", "grid.color": "#ede9df",
    "legend.facecolor": "#f5f2eb", "legend.edgecolor": "#d4cfc4",
    "lines.linewidth": 2.2, "font.family": "monospace",
    "axes.spines.top": False, "axes.spines.right": False,
})

init_db()
if "graphs_data" not in st.session_state:
    st.session_state.graphs_data = []

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">MAXCUT LAB</div>
        <div class="sidebar-brand-subtitle">Algorithm Analysis Platform</div>
        <div class="sidebar-badge">RESEARCH EDITION</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-nav-label">Navigation</div>', unsafe_allow_html=True)
    menu = st.selectbox("", ["Generate Graphs", "Analytics"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#3a4558;line-height:1.9;">
    ALGORITHMS<br>
    <span style="color:#7aabce">◆</span> Brute-Force Exact (n ≤ {BRUTE_FORCE_LIMIT})<br>
    <span style="color:#5aaa80">◆</span> Greedy Heuristic (n ≤ 100)<br><br>
    For n &gt; {BRUTE_FORCE_LIMIT}, brute-force runtime<br>is estimated via calibration.
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════
# GENERATE GRAPHS
# ═════════════════════════════════════════════

if menu == "Generate Graphs":

    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Graph Theory · NP-Hard Problems</div>
        <div class="hero-title">Maximum Cut<br><em>Problem Lab</em></div>
        <div class="hero-rule"></div>
        <div class="hero-desc">
            Explore MaxCut across randomly generated graphs from 10 to 100 nodes.
            Brute-force is exact for |V| ≤ 20; for larger graphs an estimated runtime
            is shown and only the greedy partition is computed.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="control-panel"><div class="control-panel-title">Experiment Parameters</div></div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        nodes = st.slider("Node Count  |V|", 10, 100, 15)
        if nodes <= BRUTE_FORCE_LIMIT:
            pill_cls  = "complexity-feasible"
            pill_text = f"Brute-force feasible — 2^{nodes} = {2**nodes:,} states"
        elif nodes <= 26:
            pill_cls  = "complexity-warn"
            pill_text = f"Slow — 2^{nodes} ≈ {2**nodes:,.0f} states — estimate only"
        else:
            pill_cls  = "complexity-danger"
            pill_text = f"Infeasible — 2^{nodes} states — estimate only"
        st.markdown(f'<div class="complexity-pill {pill_cls}">{pill_text}</div>', unsafe_allow_html=True)
    with col_b:
        p = st.slider("Edge Probability  p", 0.1, 1.0, 0.5)
    with col_c:
        num_graphs = st.number_input("Number of Graphs", 1, 20, 3)

    st.markdown("")
    run = st.button("▶  Run Experiment")

    if run:
        st.session_state.graphs_data = []
        bar = st.progress(0, text="Initialising…")
        for idx in range(num_graphs):
            bar.progress((idx + 1) / num_graphs, text=f"Processing graph {idx+1} of {num_graphs}…")
            G = generate_graph(nodes, p)
            brute_cut, brute_partition, brute_time, brute_estimated = brute_force_maxcut(G)
            greedy_cut, greedy_partition, greedy_time = greedy_maxcut(G)
            approx = (greedy_cut / brute_cut) if (not brute_estimated and brute_cut and brute_cut > 0) else None
            if not brute_estimated:
                insert_experiment((nodes, len(G.edges()), brute_cut, greedy_cut, brute_time, greedy_time, approx))
            st.session_state.graphs_data.append({
                "G": G, "brute_cut": brute_cut, "greedy_cut": greedy_cut,
                "brute_partition": brute_partition, "greedy_partition": greedy_partition,
                "approx": approx, "brute_time": brute_time,
                "brute_estimated": brute_estimated, "greedy_time": greedy_time,
            })
        bar.empty()

    # ── Render each graph ──
    for i, data in enumerate(st.session_state.graphs_data):
        G = data["G"]
        brute_cut        = data["brute_cut"]
        greedy_cut       = data["greedy_cut"]
        brute_partition  = data["brute_partition"]
        greedy_partition = data["greedy_partition"]
        approx           = data["approx"]
        brute_time       = data["brute_time"]
        brute_estimated  = data["brute_estimated"]
        greedy_time      = data["greedy_time"]
        n_nodes = len(G.nodes())
        n_edges = len(G.edges())

        st.markdown(f"""
        <div class="sec-head">
            <span class="sec-head-num">G{i+1:02d}</span>
            <span class="sec-head-text">Graph Instance · {n_nodes} nodes · {n_edges} edges</span>
        </div>""", unsafe_allow_html=True)

        node_size = max(60, 500 - n_nodes * 4)
        font_size = max(5, 9 - n_nodes // 20)
        fig_h     = 4.5 if n_nodes <= 30 else 5.5
        try:
            pos = nx.kamada_kawai_layout(G) if n_nodes > 20 else nx.spring_layout(G, seed=42+i)
        except Exception:
            pos = nx.spring_layout(G, seed=42+i)

        left, right = st.columns([1.35, 1])

        with left:
            st.markdown('<div class="graph-card"><div class="graph-card-tag">Base Graph — Erdős–Rényi G(n,p)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="graph-card-num">{i+1:02d}</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, fig_h))
            nx.draw_networkx(G, pos, ax=ax,
                node_color="#1a3a5c", node_size=node_size,
                edge_color="#c8bfa8", font_color="#f5f2eb",
                font_family="monospace", font_size=font_size, font_weight="bold",
                width=0.8 if n_nodes > 40 else 1.4,
                with_labels=(n_nodes <= 50),
            )
            ax.axis("off"); fig.tight_layout(pad=0.3)
            st.pyplot(fig, use_container_width=False)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            if brute_estimated:
                est_str = format_duration(brute_time)
                st.markdown(f"""
                <div class="bf-warning">
                    <div class="bf-warning-title">⚠ Brute-Force Skipped — n = {n_nodes} &gt; {BRUTE_FORCE_LIMIT}</div>
                    <div class="bf-warning-body">
                        Exact enumeration requires iterating over <strong>2<sup>{n_nodes}</sup></strong> possible partitions.
                        Based on a calibration run on this machine, that would take approximately:
                        <div class="bf-time-badge">≈ {est_str}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.metric("Optimal Cut", "N/A")
                m2.metric("Greedy Cut", greedy_cut)
                m3, m4 = st.columns(2)
                m3.metric("Brute Est. Runtime", est_str)
                m4.metric("Greedy Time", f"{greedy_time*1000:.2f} ms")
                st.markdown("""
                <div class="ratio-card" style="margin-top:0.6rem;">
                    <div class="ratio-label">Approximation Ratio  ρ</div>
                    <div class="ratio-value" style="color:#5a6480;">— unavailable</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#3a4558;margin-top:0.5rem;">
                        Requires optimal cut value — not computable for n &gt; 20
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                m1, m2 = st.columns(2)
                m1.metric("Optimal Cut", brute_cut)
                m2.metric("Greedy Cut", greedy_cut)
                m3, m4 = st.columns(2)
                m3.metric("Brute Time", f"{brute_time*1000:.2f} ms")
                m4.metric("Greedy Time", f"{greedy_time*1000:.3f} ms")
                pct = min(approx, 1.0) * 100
                grad = ('linear-gradient(90deg,#7aabce,#1a6b4a)' if approx >= 0.85
                        else 'linear-gradient(90deg,#c0392b,#b8860b)')
                st.markdown(f"""
                <div class="ratio-card">
                    <div class="ratio-label">Approximation Ratio  ρ = Greedy / Optimal</div>
                    <div class="ratio-value">{approx:.4f}</div>
                    <div class="ratio-bar-wrap"><div class="ratio-bar-fill" style="width:{pct:.1f}%;background:{grad};"></div></div>
                </div>""", unsafe_allow_html=True)
                quality = ("Excellent — greedy achieves near-optimal performance." if approx >= 0.9
                           else "Good — greedy performs within acceptable bounds." if approx >= 0.75
                           else "Suboptimal — notable gap vs. exact solution.")
                st.markdown(f"""
                <div class="insight-card" style="margin-top:0.8rem;">
                    <div class="insight-card-label">Analysis Note</div>
                    <div class="insight-card-body">{quality}</div>
                </div>""", unsafe_allow_html=True)

        # ── Partitions ──
        def draw_partition(G, pos, partition, colors, title, subtitle, nn):
            ns = max(60, 460 - nn * 4)
            fs = max(5, 8 - nn // 20)
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            nc = [colors[0] if partition[v] == 0 else colors[1] for v in G.nodes()]
            ce = [(u,v) for u,v in G.edges() if partition[u] != partition[v]]
            ie = [(u,v) for u,v in G.edges() if partition[u] == partition[v]]
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=ns)
            if nn <= 50:
                nx.draw_networkx_labels(G, pos, ax=ax, font_color="#f5f2eb",
                    font_family="monospace", font_size=fs, font_weight="bold")
            nx.draw_networkx_edges(G, pos, edgelist=ie, ax=ax,
                edge_color="#d4cfc4", width=0.7, style="dashed", alpha=0.4)
            nx.draw_networkx_edges(G, pos, edgelist=ce, ax=ax,
                edge_color="#c0392b", width=1.3 if nn > 40 else 2.0)
            ax.set_title(f"{title}\n{subtitle}", fontsize=7.5, color="#8a8070",
                fontfamily="monospace", pad=8)
            ax.axis("off"); fig.tight_layout(pad=0.3)
            return fig

        c1, c2 = st.columns(2)
        with c1:
            lbl = (f"◆  Optimal Cut — Unavailable (n={n_nodes})" if brute_estimated
                   else f"◆  Optimal Cut Partition  (G{i+1:02d})")
            with st.expander(lbl):
                if brute_estimated:
                    st.markdown(f"""
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#8a8070;
                                padding:1.5rem;text-align:center;border:1px dashed #d4cfc4;border-radius:6px;">
                        Not computed for n &gt; {BRUTE_FORCE_LIMIT}<br><br>
                        <span style="color:#a0600a;">Estimated: {format_duration(brute_time)}</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    fig_opt = draw_partition(G, pos, brute_partition, ["#1a3a5c","#c0392b"],
                        "Brute-Force Optimal", f"Cut = {brute_cut}  ·  Red edges cross partition", n_nodes)
                    st.pyplot(fig_opt, use_container_width=True)
                    st.markdown("""<div class="legend-row">
                        <span class="legend-txt"><span class="legend-dot" style="background:#1a3a5c"></span>Set S₀</span>
                        <span class="legend-txt"><span class="legend-dot" style="background:#c0392b"></span>Set S₁</span>
                    </div>""", unsafe_allow_html=True)

        with c2:
            with st.expander(f"◆  Greedy Cut Partition  (G{i+1:02d})"):
                fig_gr = draw_partition(G, pos, greedy_partition, ["#1a6b4a","#b8860b"],
                    "Greedy Heuristic", f"Cut = {greedy_cut}  ·  Red edges cross partition", n_nodes)
                st.pyplot(fig_gr, use_container_width=True)
                st.markdown("""<div class="legend-row">
                    <span class="legend-txt"><span class="legend-dot" style="background:#1a6b4a"></span>Set S₀</span>
                    <span class="legend-txt"><span class="legend-dot" style="background:#b8860b"></span>Set S₁</span>
                </div>""", unsafe_allow_html=True)

        st.divider()

    if not st.session_state.graphs_data:
        st.markdown("""<div style="text-align:center;padding:4rem 2rem;color:#c8bfa8;
            font-family:'IBM Plex Mono',monospace;font-size:0.75rem;letter-spacing:0.12em;">
            Configure parameters above and press <strong style="color:#1a3a5c">▶ Run Experiment</strong> to begin.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="footnote">MaxCut Lab · Research Interface · NP-Hard Algorithm Benchmarking</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════
# ANALYTICS
# ═════════════════════════════════════════════

if menu == "Analytics":

    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Empirical Results · Statistical Analysis</div>
        <div class="hero-title">Experiment<br><em>Analytics</em></div>
        <div class="hero-rule"></div>
        <div class="hero-desc">
            Aggregated results across recorded experiments (exact brute-force only, n ≤ 20).
            Analyse runtime scaling, cut quality, and greedy approximation vs. the optimum.
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if df.empty:
        st.markdown("""<div style="text-align:center;padding:4rem 2rem;color:#c8bfa8;
            font-family:'IBM Plex Mono',monospace;font-size:0.75rem;letter-spacing:0.12em;">
            No experiments recorded yet. Generate graphs with n ≤ 20 to populate analytics.
        </div>""", unsafe_allow_html=True)
        st.stop()

    avg_ratio = df["approx_ratio"].mean()
    max_nodes = df["nodes"].max()
    total_exp = len(df)
    min_ratio = df["approx_ratio"].min()

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-cell"><div class="stat-num">{total_exp}</div><div class="stat-lbl">Experiments</div></div>
        <div class="stat-cell"><div class="stat-num">{avg_ratio:.3f}</div><div class="stat-lbl">Avg Approx Ratio</div></div>
        <div class="stat-cell"><div class="stat-num">{min_ratio:.3f}</div><div class="stat-lbl">Worst Ratio</div></div>
        <div class="stat-cell"><div class="stat-num">{max_nodes}</div><div class="stat-lbl">Max Node Count</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head"><span class="sec-head-num">01</span><span class="sec-head-text">Experiment Log</span></div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

    st.markdown('<div class="sec-head"><span class="sec-head-num">02</span><span class="sec-head-text">Cut Weight Comparison · Optimal vs Greedy</span></div>', unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(df))
    ax1.bar(x-0.19, df["brute_cut"],  0.38, label="Optimal (Brute Force)", color="#1a3a5c", alpha=0.92)
    ax1.bar(x+0.19, df["greedy_cut"], 0.38, label="Greedy Heuristic",       color="#1a6b4a", alpha=0.82)
    ax1.set_xticks(x); ax1.set_xticklabels([f"E{j+1}" for j in x], fontsize=7.5)
    ax1.set_ylabel("Cut Weight", fontsize=8); ax1.legend(fontsize=8)
    ax1.set_title("Maximum Cut Weight per Experiment", fontsize=9, pad=10)
    ax1.grid(axis="y", alpha=0.4, linewidth=0.8); fig1.tight_layout()
    st.pyplot(fig1, use_container_width=True)

    st.markdown('<div class="sec-head"><span class="sec-head-num">03</span><span class="sec-head-text">Runtime Scaling · O(2ⁿ) vs O(n²)</span></div>', unsafe_allow_html=True)
    st.markdown("""<div class="insight-card"><div class="insight-card-label">Algorithmic Complexity</div>
    <div class="insight-card-body">Brute-force enumerates all 2ⁿ partitions — exponential in node count.
    The greedy heuristic assigns each vertex in polynomial time O(n²). The divergence is stark even at modest n.</div></div>""", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(9, 3.8))
    ax2.plot(df["nodes"], df["brute_time"],  label="Brute Force O(2ⁿ)", color="#c0392b",
             marker="o", markersize=5, markerfacecolor="#fff", markeredgewidth=1.5, zorder=3)
    ax2.plot(df["nodes"], df["greedy_time"], label="Greedy O(n²)",       color="#1a6b4a",
             marker="s", markersize=5, markerfacecolor="#fff", markeredgewidth=1.5, zorder=3)
    ax2.fill_between(df["nodes"], df["brute_time"],  alpha=0.07, color="#c0392b")
    ax2.fill_between(df["nodes"], df["greedy_time"], alpha=0.07, color="#1a6b4a")
    ax2.set_xlabel("Node Count |V|", fontsize=8); ax2.set_ylabel("Runtime (s)", fontsize=8)
    ax2.legend(fontsize=8); ax2.set_title("Runtime vs Graph Size", fontsize=9, pad=10)
    ax2.grid(axis="y", alpha=0.4, linewidth=0.8); fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    st.markdown('<div class="sec-head"><span class="sec-head-num">04</span><span class="sec-head-text">Approximation Ratio  ρ = Greedy / Optimal</span></div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(9, 3.2))
    ax3.plot(df["nodes"], df["approx_ratio"], color="#1a3a5c", marker="D",
             markersize=5, markerfacecolor="#fff", markeredgewidth=1.5, zorder=3)
    ax3.fill_between(df["nodes"], df["approx_ratio"], alpha=0.08, color="#1a3a5c")
    ax3.axhline(1.0,   color="#c0392b", linestyle="--", linewidth=1.2, label="Optimal (ρ=1.0)")
    ax3.axhline(0.878, color="#b8860b", linestyle=":",  linewidth=1.2, label="GW Guarantee (0.878)")
    ax3.set_ylim(0, 1.15); ax3.set_xlabel("Node Count |V|", fontsize=8)
    ax3.set_ylabel("Approximation Ratio", fontsize=8)
    ax3.legend(fontsize=8); ax3.set_title("Greedy Approximation Quality", fontsize=9, pad=10)
    ax3.grid(axis="y", alpha=0.4, linewidth=0.8); fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)

    st.markdown("""<div class="insight-card"><div class="insight-card-label">Theoretical Context</div>
    <div class="insight-card-body">The Goemans–Williamson SDP algorithm guarantees ρ ≥ 0.878 (dashed).
    Greedy can exceed this on small dense instances but lacks a worst-case proof.
    Points above the dashed line meet the GW threshold empirically.</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head"><span class="sec-head-num">05</span><span class="sec-head-text">Runtime Prediction · Extrapolation</span></div>', unsafe_allow_html=True)
    future_nodes, prediction = predict_runtime(df)
    pcols = st.columns(len(future_nodes))
    for col, n, pv in zip(pcols, future_nodes, prediction):
        col.metric(f"|V| = {n}", f"{pv:.4f} s")

    st.markdown('<div class="sec-head"><span class="sec-head-num">06</span><span class="sec-head-text">Runtime Scaling Forecast</span></div>', unsafe_allow_html=True)
    fig4 = runtime_prediction_plot(df)
    st.pyplot(fig4, use_container_width=True)

    st.markdown('<div class="footnote">MaxCut Lab · Analytics limited to exact experiments (n ≤ 20) · Larger graphs show greedy-only results</div>', unsafe_allow_html=True)