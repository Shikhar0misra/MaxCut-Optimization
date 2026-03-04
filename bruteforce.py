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

    start = time.time()
    best_cut, best_partition, all_cuts = brute_force_maxcut(G)
    end = time.time()

    print("Best Cut Value:", best_cut)
    print("Best Partition:", best_partition)
    print("Execution Time:", end-start)

    pos = nx.spring_layout(G)

    colors = ["red" if best_partition[i]==0 else "blue" for i in range(n)]

    cut_edges = [(u,v) for u,v in G.edges()
                 if best_partition[u] != best_partition[v]]

    # ---- Side by Side Visualization ----
    fig, axes = plt.subplots(1,2, figsize=(10,5))

    # Initial Graph
    nx.draw(G, pos, with_labels=True, ax=axes[0])
    axes[0].set_title("Initial Random Graph")

    # Max-Cut Graph
    nx.draw(G, pos, node_color=colors, with_labels=True, ax=axes[1])
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=2, ax=axes[1])
    axes[1].set_title("Max-Cut Partition")

    plt.show()


# ======================================================
# PART 2: Runtime Scaling
# ======================================================

node_sizes = list(range(7, 14))
runtimes = []

for size in node_sizes:

    G_temp = generate_graph(size, 0.5)

    start = time.time()
    brute_force_maxcut(G_temp)
    end = time.time()

    runtimes.append(end-start)


plt.figure()
plt.plot(node_sizes, runtimes, marker='o')
plt.title("Runtime vs Number of Nodes (Brute Force)")
plt.xlabel("Number of Nodes (n)")
plt.ylabel("Execution Time (seconds)")
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