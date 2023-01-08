import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import copy
from math import *


def hamiltonian_clustering(graph):
    # Using the hamiltonian given in topological transition in dynamic complex networkx
    c = nx.average_clustering(graph)
    return - c * len(graph.nodes)


def von_neumann_entropy(graph):
    # calculate the von neumann entropy for directed graph
    n_nodes = graph.number_of_nodes()
    summation = 0
    for pair in graph.edges():
        if (pair[1], pair[0]) in graph.edges:
            # bidirectional edges
            summation += graph.in_degree(pair[0]) / (graph.in_degree(pair[1]) * graph.out_degree(pair[0]) ** 2) \
                         + 1 / (graph.out_degree(pair[0]) * graph.out_degree(pair[1]))
        else:
            # unidirectional edges
            summation += graph.in_degree(pair[0]) / (graph.in_degree(pair[1]) * graph.out_degree(pair[0]) ** 2)
    s = 1 - (1 / n_nodes) - (0.5 / n_nodes ** 2) * summation
    return s


def generate_random_node_attributes(dim):
    # generate a set of attributes randomly for one node

    # list of attributes
    # preference: a multi-dimensional vector
    #             Characterize the similarity between nodes by taking inner product

    attributes = {'preference': np.random.rand(dim)}
    return attributes


def generate_initial_condition(n_nodes, n_incoming, dim):
    # generate a random graph for initializing

    graph = nx.DiGraph()
    for i in range(n_nodes):
        graph.add_node(i + 1)
        attributes = generate_random_node_attributes(dim)
        for attribute in attributes:
            graph.nodes[i + 1][attribute] = attributes[attribute]
    for i in range(n_nodes):
        incoming_node_list = np.random.randint(1, n_nodes, n_incoming)
        for j in incoming_node_list:
            if j != i:
                graph.add_edge(j, i)
            else:
                graph.add_edge(n_nodes, i)
    return graph


def montecarlo_evolution_step(graph, temperature):
    # Run a Monte Carlo Cycle, each node is chosen in order
    # Return the construction after the process
    n_nodes = len(graph)
    graph_tmp = graph
    for node in graph.nodes():
        incoming_nodes = [pair[0] for pair in graph.in_edges(node)]
        candidate_nodes = []
        for node_candidate in graph.nodes():
            if node_candidate != node and node_candidate not in incoming_nodes:
                candidate_nodes.append(node_candidate)
        if len(incoming_nodes) == 0 or len(candidate_nodes) == 0:
            continue
        # print(len(incoming_nodes), len(candidate_nodes))
        graph_new = copy.deepcopy(graph_tmp)
        removed_position = np.random.randint(0, len(incoming_nodes), 1)
        removed = incoming_nodes[removed_position[0]]
        attached_position = np.random.randint(0, len(candidate_nodes), 1)
        attached = candidate_nodes[attached_position[0]]
        graph_new.remove_edge(removed, node)
        graph_new.add_edge(attached, node)
        energy_original = hamiltonian_clustering(graph_tmp)
        energy_new = hamiltonian_clustering(graph_new)
        if energy_new < energy_original:
            graph_tmp = graph_new
        else:
            p_transition = exp((energy_original - energy_new) / temperature)
            p = np.random.rand()
            if p < p_transition:
                graph_tmp = graph_new
        del graph_new
    return graph_tmp


def montecarlo_iterate_equilibrium(n_nodes, n_incoming, dim, temperature, n_iterate):
    # Run Monte Carlo iteration in order to approach thermal equilibrium
    # with fixed iteration steps
    graph = generate_initial_condition(n_nodes, n_incoming, dim)
    energy_evolution = [hamiltonian_clustering(graph)]
    entropy_evolution = [von_neumann_entropy(graph)]
    for i in range(n_iterate):
        print('Monte Carlo steps:', i)
        graph = montecarlo_evolution_step(graph, temperature)
        energy_evolution.append(hamiltonian_clustering(graph))
        entropy_evolution.append(von_neumann_entropy(graph))
    return energy_evolution, entropy_evolution


def montecarlo_temperature_series(n_nodes, n_incoming, dim, n_iterate, t_low, t_high, t_steps):
    # Run simulation for several temperature points
    t_step = (t_high - t_low) / t_steps
    temperature_series = t_low + t_step * np.array(range(t_steps + 1))
    energy_temperature_series = [0.0 for _ in range(t_steps + 1)]
    entropy_temperature_series = [0.0 for _ in range(t_steps + 1)]
    for i in range(t_steps + 1):
        print('Temperature step:', i, 'of', t_steps)
        temperature = temperature_series[i]
        energy_evolution, entropy_evolution = montecarlo_iterate_equilibrium(n_nodes, n_incoming, dim, temperature, n_iterate)
        energy_temperature_series[i] = energy_evolution[-1]
        entropy_temperature_series[i] = entropy_evolution[-1]
    print('Energy:', energy_temperature_series)
    print('Entropy:', entropy_temperature_series)

    # Visualization
    plt.subplot(121)
    plt.plot(temperature_series, energy_temperature_series)
    plt.ylabel('Energy')
    plt.xlabel('Temperature')

    plt.subplot(122)
    plt.plot(temperature_series, entropy_temperature_series, 'r-')
    plt.ylabel('Entropy')
    plt.xlabel('Temperature')
    plt.show()
    return


# E_series, Entropy_series = montecarlo_iterate_equilibrium(100, 10, 1, 0.025, 100)
# # print(E_series)
# # print(Entropy_series)
# fig1 = plt.figure()
# plt.plot(range(len(E_series)), E_series)
# plt.ylabel('Energy')
# plt.xlabel('Monte Carlo steps')
# plt.show()
#
# fig2 = plt.figure()
# plt.plot(range(len(Entropy_series)), Entropy_series, 'r-')
# plt.ylabel('Entropy')
# plt.xlabel('Monte Carlo steps')
# plt.show()

montecarlo_temperature_series(100, 10, 1, 2000, 0.015, 0.025, 10)
