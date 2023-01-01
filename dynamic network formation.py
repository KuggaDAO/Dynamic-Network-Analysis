import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cpnet


class Node:
    # definition of Node in the graph
    def __init__(self, dim=2, t=2, order=0):
        self.preference = np.random.rand(dim)
        self.num_link = 10 + np.random.randint(0, 2)
        self.t_add = t
        self.t_order = order

    def __str__(self):
        return ''.join([str(self.t_add), '_', str(self.t_order)])


def step_nodes_num(graph, t):
    # calculate the number of nodes added to the graph at a specific time step
    num = 1
    return num


def generate_nodes_step(graph, t, dim):
    # generate a set of nodes to be added to the graph at a specific time step
    num = step_nodes_num(graph, t)
    nodes_set = []
    for i in range(num):
        new_node = Node(dim, t, i + 1)
        nodes_set.append(new_node)
    return nodes_set


def select_proportion(nodes_set, probability, num):
    # select a specific number of nodes in the given nodes set under a given probability distribution
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
    # insert a given set of nodes to graph
    # obeying the assumptions of extended scale-free model
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


def generate_random_graph(num_nodes, probability, dim):
    # generate a random undirected graph mainly for initializing
    g = nx.Graph()
    for i in range(num_nodes):
        new_node = Node(dim, -1, i)
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
    # target function for degree distribution fitting
    return a * x ** (-b)


def fit_degree_distribution(x_group, y_group):
    # fitting process for degree distribution
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


def cp_detection_discrete(graph):
    # detect discrete core-periphery structure using BE algorithm
    alg = cpnet.BE()
    alg.detect(graph)
    core_judge_address = alg.get_coreness()
    core_belong_address = alg.get_pair_id()
    core_judge_str = {}
    core_belong_str = {}
    # translate the key from address to str and find out the core nodes
    count = 0
    core_nodes_str = []
    core_nodes_address = []
    for node in graph:
        core_judge_str[node.__str__()] = core_judge_address[node]
        core_belong_str[node.__str__()] = core_belong_address[node]
        if core_judge_address[node] == 1:
            count = count + 1
            core_nodes_str.append(node.__str__())
            core_nodes_address.append(node)

    plt.subplot(111)
    ax = plt.gca()
    cpnet.draw(graph, core_belong_address, core_judge_address, ax)
    plt.title(''.join(['core nodes number:', str(count)]))
    plt.show()

    # significant test
    sig_c, sig_x, significant, p = cpnet.qstest(core_belong_address, core_judge_address, graph, alg)
    print('significant:', significant)
    print('p value:', p)
    plt.subplot(111)
    ax = plt.gca()
    # cpnet.draw(graph, sig_c, sig_x, ax)
    plt.show()
    return core_nodes_str, core_nodes_address


def cp_detection_continuous(graph):
    # detect continuous core-periphery structure using Miners algorithm
    alg = cpnet.MINRES()
    alg.detect(graph)
    core_extent_address = alg.get_coreness()
    core_belong_address = alg.get_pair_id()
    core_extent_str = {}
    core_belong_str = {}
    # translate the key from address to str and find out the core nodes
    count = 0
    core_nodes_str = []
    core_nodes_address = []
    core_extent_statistics = [0 for i in range(100)]
    for node in graph:
        core_extent_str[node.__str__()] = core_extent_address[node]
        core_belong_str[node.__str__()] = core_belong_address[node]
        core_extent_statistics[int(core_extent_address[node] / 0.01)] += 1
        if core_extent_address[node] > 0.5:
            count = count + 1
            core_nodes_str.append(node.__str__())
            core_nodes_address.append(node)
    print('core nodes:', core_nodes_str)
    plt.subplot(121)
    ax = plt.gca()
    cpnet.draw(graph, core_belong_address, core_extent_address, ax, draw_nodes_kwd={'node_size': 20})
    plt.title('graph')

    plt.subplot(122)
    plt.plot(0.01 * np.array(range(100)), core_extent_statistics)
    plt.title('Coreness statistics')
    plt.show()

    sig_c, sig_x, significant, p = cpnet.qstest(core_belong_address, core_extent_address, graph, alg)
    print('significant:', significant)
    print('p value:', p)
    plt.subplot(111)
    ax = plt.gca()
    cpnet.draw(graph, sig_c, sig_x, ax)
    plt.show()
    return


def cp_detection_multiple(graph):
    alg = cpnet.KM_config()
    alg.detect(graph)
    core_extent_address = alg.get_coreness()
    core_belong_address = alg.get_pair_id()
    core_extent_str = {}
    core_belong_str = {}
    for node in graph:
        core_extent_str[node.__str__()] = core_extent_address[node]
        core_belong_str[node.__str__()] = core_belong_address[node]
    print(core_extent_str)
    print(core_belong_str)
    plt.subplot(111)
    ax = plt.gca()
    cpnet.draw(graph, core_belong_address, core_extent_address, ax, draw_nodes_kwd={'node_size': 20})
    plt.show()

    # significant test
    sig_c, sig_x, significant, p = cpnet.qstest(core_belong_address, core_extent_address, graph, alg)
    print('significant:', significant)
    print('p value:', p)
    plt.subplot(111)
    ax = plt.gca()
    cpnet.draw(graph, sig_c, sig_x, ax)
    plt.show()
    return



dimension = 2
G = generate_random_graph(20, 0.3, dimension)
t_step = 1000
for t_add in range(t_step):
    if t_add % 100 == 0:
        print(t_add, 'time steps ended')
    new_nodes_set = generate_nodes_step(G, t_add, dimension)
    insert_node_step(G, new_nodes_set)

# calculate clustering coefficient
cluster_coef = nx.average_clustering(G)
print('Clustering Coefficient:', cluster_coef)

# small_world_coef = nx.sigma(G)
# print(small_world_coef)

# calculate the average_shortest path
# ave_shortest_l = nx.average_shortest_path_length(G)
# print('Average shortest length:', ave_shortest_l)

# use power law function to fit the degree distribution
distribution = nx.degree_histogram(G)
X_fit, Y_fit, A, Gamma = fit_degree_distribution(range(len(distribution)), distribution)
print('Gamma:', Gamma)

# core-periphery structure
# c_str, c_address = cp_detection_discrete(G)
# print(c_str)
cp_detection_continuous(G)
# cp_detection_multiple(G)

# Visualize
# plt.subplot(111)
# plt.plot(range(len(distribution)), distribution)
# plt.plot(X_fit, Y_fit)
# plt.title(' '.join(['Degree distribution, Clustering Coefficient:', str(round(cluster_coef, 4)),
#                     'Average shortest length:', str(round(ave_shortest_l, 4)), 'Gamma:', str(round(Gamma, 4))]))

# plt.subplot(122)
# options = {
#     'node_size': 100,
#     'font_size': 6,
# }
#
# nx.draw_networkx(G, **options)
# plt.title(' '.join(['Cluster Coefficient:', str(cluster_coef)]))

plt.show()





