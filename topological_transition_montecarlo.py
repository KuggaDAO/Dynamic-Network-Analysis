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


def hamiltonian_degree_linking(graph):
    # Calculate the Hamiltonian defined on edges
    # The contribution of an edge to the hamiltonian is

    summation = 0
    for pair in graph.edges:
        summation += 1 / (graph.degree(pair[0]) * graph.degree(pair[1]))
    return summation


def hamiltonian_degree_linking_local(graph, node_selected, node_old, node_new):
    # Calculate part of the Hamiltonian regarding the three nodes
    energy_old = 0
    energy_new = 0
    for pair in graph.out_edges(node_old):
        energy_old += 1 / (graph.degree(pair[0]) * graph.degree(pair[1]))
        if pair[1] != node_selected:
            energy_new += 1 / ((graph.degree(pair[0]) - 1) * graph.degree(pair[1]))
    for pair in graph.out_edges(node_new):
        energy_old += 1 / (graph.degree(pair[0]) * graph.degree(pair[1]))
        energy_new += 1 / ((graph.degree(pair[0]) + 1) * graph.degree(pair[1]))
    energy_new += 1 / (graph.degree(node_selected) * (graph.degree(node_new) + 1))
    for pair in graph.in_edges(node_old):
        energy_old += 1 / (graph.degree(pair[0]) * graph.degree(pair[1]))
        energy_new += 1 / (graph.degree(pair[0]) * (graph.degree(pair[1]) - 1))
    for pair in graph.in_edges(node_new):
        energy_old += 1 / (graph.degree(pair[0]) * graph.degree(pair[1]))
        energy_new += 1 / (graph.degree(pair[0]) * (graph.degree(pair[1]) + 1))
    return energy_old, energy_new


def von_neumann_entropy(graph):
    # calculate the von neumann entropy for directed graph
    n_nodes = graph.number_of_nodes()
    summation = 0
    for pair in graph.edges:
        if (pair[1], pair[0]) in graph.edges:
            # bidirectional edges
            summation += graph.in_degree(pair[0]) / (graph.in_degree(pair[1]) * graph.out_degree(pair[0]) ** 2) \
                         + 1 / (graph.out_degree(pair[0]) * graph.out_degree(pair[1]))
        else:
            # unidirectional edges
            summation += graph.in_degree(pair[0]) / (graph.in_degree(pair[1]) * graph.out_degree(pair[0]) ** 2)
    s = 1 - (1 / n_nodes) - (0.5 / n_nodes ** 2) * summation
    return s


def zero_out_degree_percentage(graph):
    total = 0
    zero_out_degree = 0
    for node in graph:
        total += 1
        if graph.out_degree(node) == 0:
            zero_out_degree += 1
    return zero_out_degree / total


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
    for i in range(1, n_nodes + 1):
        incoming_node_list = np.random.choice(range(1, n_nodes), size=n_incoming, replace=False)
        for j in incoming_node_list:
            if j != i:
                graph.add_edge(j, i)
            else:
                graph.add_edge(n_nodes, i)
    return graph


def generate_stable_initial_condition(n_nodes, n_incoming, dim):
    # generate a fixed graph for initializing
    graph = nx.DiGraph()
    for i in range(n_nodes):
        graph.add_node(i + 1)
        attributes = generate_random_node_attributes(dim)
        for attribute in attributes:
            graph.nodes[i + 1][attribute] = attributes[attribute]
    for i in range(1, n_incoming + 1):
        for j in range(1, n_incoming + 1):
            if j == i:
                graph.add_edge(n_incoming + 1, i)
            else:
                graph.add_edge(j, i)
    for i in range(n_incoming + 1, n_nodes + 1):
        for j in range(1, n_incoming + 1):
            graph.add_edge(j, i)
    return graph


def montecarlo_evolution_step(graph, temperature):
    # Run a Monte Carlo Cycle, each node is chosen in order
    # Return the construction after the process
    n_nodes = len(graph)
    graph_tmp = graph
    for node in graph.nodes():
        incoming_nodes = [pair[0] for pair in graph_tmp.in_edges(node)]
        candidate_nodes = []
        for node_candidate in graph_tmp.nodes():
            if node_candidate != node and node_candidate not in incoming_nodes:
                candidate_nodes.append(node_candidate)
        if len(incoming_nodes) != graph_tmp.in_degree(node) or len(candidate_nodes) == 0:
            print('error')
            continue
        # print(len(incoming_nodes), len(candidate_nodes))
        graph_new = copy.deepcopy(graph_tmp)
        removed_position = np.random.randint(0, len(incoming_nodes), 1)
        removed = incoming_nodes[removed_position[0]]
        attached_position = np.random.randint(0, len(candidate_nodes), 1)
        attached = candidate_nodes[attached_position[0]]
        graph_new.remove_edge(removed, node)
        graph_new.add_edge(attached, node)
        # energy_original = hamiltonian_clustering(graph_tmp)
        # energy_new = hamiltonian_clustering(graph_new)
        energy_original, energy_new = hamiltonian_degree_linking_local(graph_tmp, node, removed, attached)
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
    # graph = generate_stable_initial_condition(n_nodes, n_incoming, dim)
    # energy_evolution = [hamiltonian_clustering(graph)]
    energy_evolution = [hamiltonian_degree_linking(graph)]
    entropy_evolution = [von_neumann_entropy(graph)]
    zero_percentage = [zero_out_degree_percentage(graph)]
    for i in range(n_iterate):
        print('Monte Carlo steps:', i)
        graph = montecarlo_evolution_step(graph, temperature)
        # check module
        # for node in graph:
        #     if graph.in_degree(node) != n_incoming:
        #         print('error:', node, graph.in_degree(node))
        # energy_evolution.append(hamiltonian_clustering(graph))
        energy_evolution.append(hamiltonian_degree_linking(graph))
        entropy_evolution.append(von_neumann_entropy(graph))
        zero_percentage.append(zero_out_degree_percentage(graph))
    # visualize the resulting network
    nx.draw_networkx(graph)
    plt.title(''.join(['T=', str(round(temperature, 6)), ' E=', str(round(energy_evolution[-1], 4)), ' S=', str(round(entropy_evolution[-1], 6))]))
    plt.show()
    return energy_evolution, entropy_evolution, zero_percentage


def montecarlo_temperature_series(n_nodes, n_incoming, dim, n_iterate, t_low, t_high, t_steps):
    # Run simulation for several temperature points
    t_step = (t_high - t_low) / t_steps
    temperature_series = t_low + t_step * np.array(range(t_steps + 1))
    energy_temperature_series = [0.0 for _ in range(t_steps + 1)]
    entropy_temperature_series = [0.0 for _ in range(t_steps + 1)]
    zero_percentage_series = [0.0 for _ in range(t_steps + 1)]
    for i in range(t_steps + 1):
        print('Temperature step:', i, 'of', t_steps)
        temperature = temperature_series[i]
        energy_evolution, entropy_evolution, percentage_evolution = \
            montecarlo_iterate_equilibrium(n_nodes, n_incoming, dim, temperature, n_iterate)
        energy_temperature_series[i] = np.mean(energy_evolution[-10:])
        print(energy_temperature_series[i])
        entropy_temperature_series[i] = np.mean(entropy_evolution[-10:])
        print(entropy_temperature_series[i])
        zero_percentage_series[i] = percentage_evolution[-1]
        print(zero_percentage_series[i])
    # print('Energy:', energy_temperature_series)
    # print('Entropy:', entropy_temperature_series)

    # Visualization
    plt.subplot(211)
    plt.plot(temperature_series, energy_temperature_series)
    plt.ylabel('Energy')
    plt.xlabel('Temperature')

    plt.subplot(223)
    plt.plot(temperature_series, entropy_temperature_series, 'r-')
    plt.ylabel('Entropy')
    plt.xlabel('Temperature')

    plt.subplot(224)
    plt.plot(temperature_series, zero_percentage_series, 'g-')
    plt.ylabel('0 Out Degree Percentage')
    plt.xlabel('Temperature')
    plt.show()
    return


E_series, Entropy_series, Percentage_series = montecarlo_iterate_equilibrium(100, 10, 1, 0.001, 2000)
# print(E_series)
# print(Entropy_series)
fig1 = plt.figure()
plt.plot(range(len(E_series)), E_series)
plt.ylabel('Energy')
plt.xlabel('Monte Carlo steps')
plt.show()

fig2 = plt.figure()
plt.plot(range(len(Entropy_series)), Entropy_series, 'r-')
plt.ylabel('Entropy')
plt.xlabel('Monte Carlo steps')
plt.show()

# montecarlo_temperature_series(100, 10, 1, 2000, 0.0001, 0.0011, 50)
