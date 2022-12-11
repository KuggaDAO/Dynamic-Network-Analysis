import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


class Node:
    def __init__(self, dim=2, t=0):
        self.preference = np.random.rand(dim)
        self.num_link = 10 + np.random.randint(0, 2)
        self.t_add = t

    def __str__(self):
        return str(self.t_add)


def step_nodes_num(graph, t):
    num = 1
    return num


def generate_nodes_step(graph, t):
    num = step_nodes_num(graph, t)
    nodes_set = []
    for i in range(num):
        new_node = Node(2, t)
        nodes_set.append(new_node)
    return nodes_set


def select_proportion(nodes_set, probability, num):
    candidates = [node for node in nodes_set]
    p = [x for x in probability]
    chosen = []
    for i in range(num):
        s = sum(p)
        position = s * np.random.rand()
        part_s = 0
        num_chosen = -1
        while part_s < position:
            num_chosen = num_chosen + 1
            part_s = part_s + p[num_chosen]
        chosen.append(candidates[num_chosen])
        candidates.pop(num_chosen)
        p.pop(num_chosen)
    return chosen


def insert_node_step(graph, nodes_set):
    link_to_nodes = []
    for node_x in nodes_set:
        total = 0
        for node_exist in graph.nodes():
            total = total + graph.degree(node_exist) * np.dot(node_x.preference, node_exist.preference)
        p = []
        for node_exist in graph.nodes():
            p.append(graph.degree(node_exist) * np.dot(node_x.preference, node_exist.preference) / total)
        link_to_nodes.append(select_proportion(graph.nodes(), p, node_x.num_link))
    graph.add_nodes_from(nodes_set)
    for i in range(len(nodes_set)):
        for j in range(len(link_to_nodes[i])):
            graph.add_edge(nodes_set[i], link_to_nodes[i][j])
    return


def generate_random_graph(num_nodes, probability):
    g = nx.Graph()
    for i in range(num_nodes):
        new_node = Node(2, -1)
        count = 0
        link_nodes = []
        for node_exist in g.nodes():
            p = np.random.rand()
            if p < probability:
                link_nodes.append(node_exist)
                count = count + 1
        new_node.num_link = count
        g.add_node(new_node)
        for node_exist in link_nodes:
            g.add_edge(new_node, node_exist)
    return g


def f_1(x, a, b):
    return a * x ** (-b)


def fit_degree_distribution(x_group, y_group):
    flag = 0
    xs = [x for x in x_group]
    ys = [y for y in y_group]
    while flag == 0:
        if ys[0] < 8:
            xs.pop(0)
            ys.pop(0)
        else:
            flag = 1
    a, b = sp.optimize.curve_fit(f_1, xs, ys)[0]
    y_fit = [f_1(x, a, b) for x in xs]
    return xs, y_fit, a, b


G = generate_random_graph(20, 0.5)
t_step = 100
for t_add in range(t_step):
    new_nodes_set = generate_nodes_step(G, t_add)
    insert_node_step(G, new_nodes_set)

cluster_coef = nx.average_clustering(G)
print('Clustering Coefficient:', cluster_coef)

# small_world_coef = nx.sigma(G)
# print(small_world_coef)

ave_shortest_l = nx.average_shortest_path_length(G)
print('Average shortest length:', ave_shortest_l)

distribution = nx.degree_histogram(G)
X_fit, Y_fit, A, Gamma = fit_degree_distribution(range(len(distribution)), distribution)
print('Gamma:', Gamma)

# Visualize
plt.subplot(111)
plt.plot(range(len(distribution)), distribution)
plt.plot(X_fit, Y_fit)
plt.title(' '.join(['Degree distribution, Clustering Coefficient:', str(round(cluster_coef, 4)),
                    'Average shortest length:', str(round(ave_shortest_l, 4)), 'Gamma:', str(round(Gamma, 4))]))

# plt.subplot(122)
# options = {
#     'node_size': 100,
#     'font_size': 6,
# }
#
# nx.draw_networkx(G, **options)
# plt.title(' '.join(['Cluster Coefficient:', str(cluster_coef)]))

plt.show()





