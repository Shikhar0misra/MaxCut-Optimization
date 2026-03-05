import itertools
import random
import time
import networkx as nx
import matplotlib.pyplot as plt


# ---------------------------
# Generate Erdős–Rényi Graph
# ---------------------------
def generate_graph(n, p, seed=None):
    if seed is not None:
        random.seed(seed)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                G.add_edge(i, j)

    return G


# ---------------------------
# Compute Cut Value
# ---------------------------
def compute_cut(G, partition):
    cut_value = 0
    for u, v in G.edges():
        if partition[u] != partition[v]:
            cut_value += 1
    return cut_value


# ---------------------------
# Brute Force Max-Cut
# ---------------------------
def brute_force_maxcut(G):

    n = len(G.nodes())

    best_cut = 0
    best_partition = None
    all_cuts = []

    for bits in itertools.product([0,1], repeat=n):

        cut = compute_cut(G, bits)
        all_cuts.append(cut)

        if cut > best_cut:
            best_cut = cut
            best_partition = bits

    return best_cut, best_partition, all_cuts


# ---------------------------
# Greedy / Local Search Max-Cut
# ---------------------------
def greedy_maxcut(G):

    n = len(G.nodes())

    # Start with random partition
    partition = [random.randint(0,1) for _ in range(n)]

    improved = True

    while improved:

        improved = False

        for node in range(n):

            current_cut = compute_cut(G, partition)

            # Flip node
            partition[node] = 1 - partition[node]

            new_cut = compute_cut(G, partition)

            if new_cut > current_cut:
                improved = True
            else:
                # Revert flip
                partition[node] = 1 - partition[node]

    best_cut = compute_cut(G, partition)

    return best_cut, partition


# ======================================================
# PART 1: Demonstrate Multiple Graphs (n = 7)
# ======================================================

n = 7
p = 0.5
num_examples = 3

for example in range(num_examples):

    print("\n==============================")
    print("Example Graph", example+1)
    print("==============================")

    G = generate_graph(n, p)

    # ---- Brute Force ----
    start = time.time()
    best_cut, best_partition, all_cuts = brute_force_maxcut(G)
    end = time.time()

    print("Brute Force Cut Value:", best_cut)
    print("Brute Partition:", best_partition)
    print("Brute Execution Time:", end-start)

    # ---- Greedy ----
    start = time.time()
    greedy_cut, greedy_partition = greedy_maxcut(G)
    end = time.time()

    print("Greedy Cut Value:", greedy_cut)
    print("Greedy Partition:", greedy_partition)
    print("Greedy Execution Time:", end-start)

    pos = nx.spring_layout(G)

    brute_colors = ["red" if best_partition[i]==0 else "blue" for i in range(n)]
    greedy_colors = ["red" if greedy_partition[i]==0 else "blue" for i in range(n)]

    brute_cut_edges = [(u,v) for u,v in G.edges()
                 if best_partition[u] != best_partition[v]]

    greedy_cut_edges = [(u,v) for u,v in G.edges()
                 if greedy_partition[u] != greedy_partition[v]]

    # ---- Side by Side Visualization (3 graphs) ----
    fig, axes = plt.subplots(1,3, figsize=(15,5))

    # Initial Graph
    nx.draw(G, pos, with_labels=True, ax=axes[0])
    axes[0].set_title("Initial Random Graph")

    # Brute Force Result
    nx.draw(G, pos, node_color=brute_colors, with_labels=True, ax=axes[1])
    nx.draw_networkx_edges(G, pos, edgelist=brute_cut_edges, width=2, ax=axes[1])
    axes[1].set_title("Brute Force Max-Cut")

    # Greedy Result
    nx.draw(G, pos, node_color=greedy_colors, with_labels=True, ax=axes[2])
    nx.draw_networkx_edges(G, pos, edgelist=greedy_cut_edges, width=2, ax=axes[2])
    axes[2].set_title("Greedy Max-Cut")

    plt.show()


# ======================================================
# PART 2: Runtime Scaling
# ======================================================

node_sizes = list(range(7, 14))
brute_runtimes = []
greedy_runtimes = []

for size in node_sizes:

    G_temp = generate_graph(size, 0.5)

    # Brute Force
    start = time.time()
    brute_force_maxcut(G_temp)
    end = time.time()
    brute_runtimes.append(end-start)

    # Greedy
    start = time.time()
    greedy_maxcut(G_temp)
    end = time.time()
    greedy_runtimes.append(end-start)


plt.figure()
plt.plot(node_sizes, brute_runtimes, marker='o', label="Brute Force")
plt.plot(node_sizes, greedy_runtimes, marker='o', label="Greedy")
plt.title("Runtime vs Number of Nodes")
plt.xlabel("Number of Nodes (n)")
plt.ylabel("Execution Time (seconds)")
plt.legend()
plt.show()


# ======================================================
# PART 3: Exponential Growth of Configurations
# ======================================================

config_counts = [2**n for n in node_sizes]

plt.figure()
plt.plot(node_sizes, config_counts, marker='o')
plt.title("Number of Configurations (2^n) vs Nodes")
plt.xlabel("Number of Nodes (n)")
plt.ylabel("Total Configurations (2^n)")
plt.show()


# ======================================================
# PART 4: All 128 Partitions Visualization
# ======================================================

G = generate_graph(7,0.5)

best_cut, best_partition, all_cuts = brute_force_maxcut(G)

plt.figure()
plt.bar(range(len(all_cuts)), all_cuts)
plt.title("All 128 Partitions vs Cut Value (n=7)")
plt.xlabel("Partition Index")
plt.ylabel("Cut Value")
plt.show()