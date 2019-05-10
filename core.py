import timeit
import pickle
import numpy as np
import scipy.stats as st
import networkx as nx
import matplotlib.pyplot as plt

from simulation.helpers import update_progress, csv_export, create_random_graph_distances
from simulation.plotting import print_graph, print_tips_over_time, \
print_tips_over_time_multiple_agents, print_tips_over_time_multiple_agents_with_tangle, \
print_attachment_probabilities_alone, print_attachment_probabilities_all_agents, print_confirmation_confidences_all_agents
from simulation.simulation import Single_Agent_Simulation
from simulation.simulation_multi_agent import Multi_Agent_Simulation

#############################################################################
# SIMULATION: SINGLE AGENT
#############################################################################

#Parameters: no_of_transactions, lambda, no_of_agents, alpha, latency (h), tip_selection_algo
#Tip selection algorithms: Choose among "random", "weighted", "unweighted" as input

# simu = Single_Agent_Simulation(100, 50, 1, 0.005, 1, "weighted")
# simu.setup()
# simu.run()
# print_tips_over_time(simu)

#############################################################################
# SIMULATION: MULTI AGENT
#############################################################################

#Parameters: no_of_transactions, lambda, no_of_agents, alpha, distance, tip_selection_algo
# latency (default value 1), agent_choice (default vlaue uniform distribution, printing)
#Tip selection algorithms: Choose among "random", "weighted", "unweighted" as input

start_time = timeit.default_timer()
runs = 1
data = []
for i in range(runs):
    run_data = []
    simu2 = Multi_Agent_Simulation(_no_of_transactions=50, _lambda=2, _no_of_agents=2, \
                 _alpha=0.5, _distance=1, _tip_selection_algos=["weighted", "lazy"], \
                 _conflict_coefficient=.35,  _printing=False)
    simu2.setup()
    simu2.run()
    print(simu2.get_heaviest_chain())
    # print(simu2.weights)
    print_tips_over_time_multiple_agents_with_tangle(simu2, simu2.no_of_transactions)
    # for agent in range(simu2.no_of_agents):
    #     agent_data = [i[1][agent] for i in simu2.record_attachment_probabilities]
    #     run_data.append(agent_data)
    print_confirmation_confidences_all_agents(simu2)
    attachment = simu2.attachment_probabilities
    # print("ATTACH", attachment)
    data.append(attachment)

data = np.mean(data, axis=0)
print("OVERALL ATTACHMENT:", data)
# print_attachment_probabilities_all_agents(simu2, data)


    # csv_export(simu2)

print("TOTAL simulation time: " + str(np.round(timeit.default_timer() - start_time, 3)) + " seconds\n")

#############################################################################
# PLOTTING
#############################################################################

# print_graph(simu2)
# print_tips_over_time(simu2)
# print_tips_over_time_multiple_agents(simu2, simu2.no_of_transactions)

