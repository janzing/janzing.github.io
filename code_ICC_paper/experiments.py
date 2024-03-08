# This is code for the experiments in the AISTATS 2024 paper
# Quantifying Intrinsic Contribution via Structure-Preserving Interventions by
# Janzing, Bloebaum, Mastakouri, Faller, Minorics, Budhathoki
#
# Copyright Dominik Janzing, 08 March 2024, creative common license  CC BY-SA 4.0


import numpy as np, pandas as pd, networkx as nx
from dowhy import gcm
from dowhy.gcm.util.general import variance_of_deviations
from itertools import permutations
import os
from sklearn.linear_model import LinearRegression
from datetime import datetime
import matplotlib.pyplot as plt

def load_river_data(nodes):
    time_stamps = pd.read_csv(os.path.join('river_data','target.csv'), usecols=[2], header=0).iloc[:,0]
    waterflows = pd.DataFrame(index=time_stamps, columns= [])
    for col in nodes:

        waterflows_add = pd.read_csv(os.path.join('river_data', col + '.csv'), index_col=0,
                                     usecols=[2,3,5], header=0)
        waterflows_add = waterflows_add[waterflows_add['quality'] == 'Good']
        waterflows_add.drop(columns=['quality'], inplace=True)
        waterflows_add.columns = [col]
        waterflows_add[col] = waterflows_add[col].astype('float64')
        waterflows = pd.concat((waterflows,waterflows_add), join='inner', axis=1)
    end_time = datetime.strptime('2021-12-11', '%Y-%m-%d')
    selected_dates = [time for time in waterflows.index if datetime.strptime(time, '%Y-%m-%d') < end_time]
    waterflows = waterflows.loc[selected_dates]
    return waterflows


def load_mpg_data():
    return pd.read_csv(os.path.join('auto mpg data', 'auto-mpg.csv'), index_col=None,
                usecols=[0,1,2,3,4], sep=';', names=['mpg','cyl','dis','hp','wgt'], header=0)

def do_shapley_river(noise, target):
    non_target_nodes = noise.columns[:-1]
    contribution = pd.Series([0]*4, index=non_target_nodes)
    def target_variance(adjustment_set):
        if 'mediator' in adjustment_set:
            return np.var(noise['target'])
        else:
            non_adjustment_set = list(set(noise.columns) - set(adjustment_set))
            return np.var(noise[non_adjustment_set].sum(axis=1))

    orderings = permutations(non_target_nodes)
    count = 0
    for ordering in orderings:
        count += 1
        ordering = list(ordering)
        for j in range(1,len(ordering)):
            contribution[ordering[j-1]] += target_variance(ordering[0:j-1]) - target_variance(ordering[0:j])
    contribution = contribution.div(count)
    return contribution/contribution.sum() * 100

def variance_shap(data, target):
    mdl = LinearRegression()
    mdl.fit(data.to_numpy(), target)
    relevance = gcm.feature_relevance_distribution(mdl.predict,
                                                   data.to_numpy(), subset_scoring_func=variance_of_deviations)
    return relevance / sum(relevance) * 100

def noise_from_known_fcm(river_flows):
    noise = pd.DataFrame(index=river_flows.index, columns=river_flows.columns)
    # noise of root nodes
    for node in nodes[0:3]:
        noise[node] = river_flows[node]
    # noise of mediator
    y = river_flows['mediator'].to_numpy()
    X = river_flows[['root1','root2','root3']].to_numpy()
    noise['mediator'] = y - X.sum(axis=1)
    # noise of target
    y = river_flows['target'].to_numpy()
    x = river_flows['mediator'].to_numpy()
    noise['target'] = y - x
    return noise

def generate_icc_plot_with_confidence(noise, river_flow_target):
    def f():
        variance_sh = variance_shap(noise, river_flow_target)
        return variance_sh/sum(variance_sh) * 100

    bootstrap_results = gcm.confidence_intervals(f)
    print('ICC values:')
    print(bootstrap_results[0])
    print('confidence intervals')
    intervals = bootstrap_results[1]
    print(intervals)
    lower_bounds = intervals[:,0]
    upper_bounds = intervals[:,1]
    number_of_nodes = 5
    for lower,upper,x in zip(lower_bounds,upper_bounds,range(number_of_nodes)):
        plt.plot((x,x), (lower,upper),'ro-',color='red')
    plt.xticks(range(number_of_nodes),['Henthorn', 'Hodder Place', 'Whalley Weir', 'New Jumbles R.', 'Samlesbury'])
    plt.ylabel('normalized ICC in %')
    plt.savefig('icc_plot')
    plt.show()


if __name__ == "__main__":

    # experiments for River data (ICC, variance-based Shapley, and variance-based do-Shapley)

    # all nodes of the DAG where 3 root nodes influence the target via the mediator node

    # preparation of data
    nodes = ['root1','root2','root3','mediator','target']
    river_flows = load_river_data(nodes)

    # causality-blind variance-based Shapley:
    print('causality-agnostic Shapley in %')
    print(variance_shap(river_flows.iloc[:,:-1], river_flows['target']))


    # pre-computation for ICC and do-Shapley:
    # compute noise from residuals of the sum
    noise = noise_from_known_fcm(river_flows)

    # variance-based do-Shapley
    print('do Shapley in %')
    print(do_shapley_river(noise, river_flows['target']))

    # ICC
    print("normalized ICC values in % and plot with confidence levels")
    generate_icc_plot_with_confidence(noise, river_flows['target'])


    # ICC for Auto mpg data
    mpg_data = load_mpg_data()
    causal_model = gcm.StructuralCausalModel(nx.DiGraph([('cyl', 'dis'), ('dis', 'wgt'),
                                                         ('dis','hp'),('wgt','mpg'),('hp','mpg')]))
    gcm.auto.assign_causal_mechanisms(causal_model, mpg_data)
    gcm.fit(causal_model, mpg_data)
    contributions = gcm.intrinsic_causal_influence(causal_model, 'mpg')
    print('normalized ICC values for auto mpg in %:')
    contributions = pd.Series(contributions)
    print(contributions/contributions.sum() * 100)

