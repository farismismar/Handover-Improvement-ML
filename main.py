#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 3 16:20:05 2017

@author: farismismar
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from matplotlib import rc
from numpy import linalg as LA

import xgboost as xgb
from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

import os
os.chdir('/Users/farismismar/Desktop/E_Projects/UT Austin Ph.D. EE/Papers/3- Partially Blind Handovers for mmWave Communications Aided by sub-6 GHz LTE Signaling/')

seed = 3
np.random.seed(seed)
event_a1_measurement_gap = -125 # dBm # close the measurement gap
event_a2_measurement_gap = -130 # dBm # open the measurement gap

RSRP_5G_min = -95 # minimum feasible communication in 5G (say?)
f_mmWave = 28e9 # 28 GHz
c = 3e8 # speed of light in m/s
NRB = 100

# r  the radius of the circle
r = 350 # in meters
lamb = 2e-4 #rate for PPP intensity parameter
r_training = 0.7 

Lambda = lamb * np.pi * r ** 2 # the mean of the Poisson random variable n
n = np.random.poisson(Lambda) # the Poisson random variable (i.e., the number of points inside C)

max_sim_time = 40 # in milliseconds, T_sim

k = 5 # 5-fold cross validationl

# Now shadowing used here
def cost231(distance, f=2.1e3, h_R=1.5, h_B=20):
    C = 3
    a = (1.1 * np.log10(f) - 0.7)*h_R - (1.56*np.log10(f) - 0.8)
    L = []
    for d in distance:
        L.append(46.3 + 33.9 * np.log10(f) + 13.82 * np.log10(h_B) - a + (44.9 - 6.55 * np.log10(h_B)) * np.log10(d) + C)

    return L

# From Path Loss Models for Millimeter-wave  distance-dependent large-scale propagation measurements and path loss models for outdoor and indoor 5G systems},'' April 2016.
def pathloss_5g(distance, f=28e3, h_R=1.5, h_B=23):
    # These are the parameters for f = 28000 MHz.
    alpha = 118.77
    beta = 0.12
    sigma_sf = 5.78
    
    L = []
    for d in distance:
        chi_sigma = np.random.normal(0, sigma_sf) # shadowing 
        L.append(alpha + beta * np.log10(d * 1000) + chi_sigma) # distance is in meters here.

    return L


def plot_network(u_1, u_2, plotting=False):
    
    n = len(u_1)  
    radii = np.zeros(n) # the radial coordinate of the points
    angle = np.zeros(n) # the angular coordinate of the points
    
    for i in range(n):
        radii[i] = r * (np.sqrt(u_1[i]))
        angle[i] = 2 * np.pi * u_2[i]    
    
     # Cartesian Coordinates
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        x[i] = radii[i] * np.cos(angle[i])
        y[i] = radii[i] * np.sin(angle[i])
    
    if (plotting):
        plt.figure(figsize=(5,5))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.plot(x,y,'bo', markersize=1.5)
        plt.plot(0, 0, '^r')
        plt.grid(True)
        
        plt.title(r'\textbf{Base station and UE positions}')
        plt.xlabel('X pos (m)')
        plt.ylabel('Y pos (m)')
        
        ax = plt.gca()
        circ = plt.Circle((0, 0), radius=r, color='r', linewidth=2, fill=False)
        ax.add_artist(circ)
        
        plt.xlim(-r, r)
        plt.ylim(-r, r)
    
        plt.savefig('figures/network.pdf', format='pdf')
        plt.show()
    
    return ([x,y])

def plot_roc(y_test, y_score, i):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7,6))
    lw = 2
    
    plt.plot(fpr, tpr,
         lw=lw, label="ROC curve (AUC = {:.6f})".format(roc_auc))
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(r'\textbf{Receiver Operating Characteristic -- UE \#' + '{0}'.format(i) + '}')
    plt.legend(loc="lower right")
    plt.savefig('figures/roc_{0}.pdf'.format(i), format='pdf')

def predict_handover(UE_i_RSRP_LTE, UE_i_RSRP_5G, UE_i_HO, i):
    dataset = pd.DataFrame({'RSRP_LTE': UE_i_RSRP_LTE.values,
                            'RSRP_5G': UE_i_RSRP_5G.values,
                            'Executed': UE_i_HO.values})
    
    # Convert flag to 1 and 0 to become a supervised label.
    dataset['Executed'] = dataset['Executed'].astype(int)
    
    training = dataset.iloc[:N_training,:]
    test = dataset.iloc[N_training:,:]
    
    X_train = training.drop('Executed', axis=1)
    y_train = training['Executed']
    X_test = test.drop('Executed', axis=1)
    y_test = test['Executed']

    classifier = xgb.XGBClassifier(seed=seed, learning_rate=0.05, n_estimators = 500)
    #classifier.get_params().keys()
    
    # Hyperparameters
    alphas = np.linspace(0,1,2)
    lambdas = np.linspace(0,1,2)
    depths = [6,8]
    sample_weights = [0.5, 0.7]
    child_weights = [0, 10]
    objectives = ['binary:logistic', 'reg:linear']
    
    hyperparameters = {'reg_alpha': alphas, 'reg_lambda': lambdas, 'objective': objectives, 'max_depth': depths, 
                       'colsample_bytree': sample_weights, 'min_child_weight': child_weights}
    #hyperparameters = {'objective': objectives}

  
    gs_xgb = GridSearchCV(classifier, hyperparameters, scoring='roc_auc', cv=k) # k-fold crossvalidation
    gs_xgb.fit(X_train, y_train)
    clf = gs_xgb.best_estimator_
    y_pred = clf.predict(X_test)

    y_score = clf.predict_proba(X_test)
    
    try:
        # Compute area under ROC curve
        roc_auc = roc_auc_score(y_test, y_score[:,1])
        
        print('The ROC AUC for this UE #{0} is {1:.6f}'.format(i, roc_auc))
    
        if (roc_auc > 0.7):
            y_pred=pd.DataFrame(y_pred)
            output = y_train.append(y_pred, ignore_index=True)
            plot_roc(y_test, y_score[:,1], i)
        else:
            output = y_train.append(y_test) # algorithm failed---use original
    except:
        print('The ROC AUC for this UE #{0} is N/A')
        output = y_train.append(y_test) # algorithm failed---use original
         
    return output.values

def compute_distance(X, Y):
    # Distances in kilometers.
    dist = LA.norm((X, Y), axis=0)
    return dist / 1000.

def compute_power(d, tx=46, g=17, loss=3, f=2100):
    # Distances in meters.
    path_loss = cost231(d, f, 1.5, 20)
    n = len(path_loss)
    return np.ones(n) * (tx + g - loss - 10*np.log10(NRB * 12)) - path_loss

def compute_power_5g(d, tx=46, g=24, loss=3, f=(f_mmWave / 1e6)): # need to find the proper URA antenna gain.
    NRB = 200
    # Distances in meters.
    path_loss = pathloss_5g(d, f, 1.5, 20)
    n = len(path_loss)
    return np.ones(n) * (tx + g - loss - 10*np.log10(NRB * 12)) - path_loss

u_1 = np.random.uniform(0.0, 1.0, n) # generate n uniformly distributed points 
u_2 = np.random.uniform(0.0, 1.0, n) # generate another n uniformly distributed points 
  
([x0, y0]) = plot_network(u_1, u_2, plotting=True)#r=r, C=C)


simulation_data = pd.DataFrame({'X': x0,
                                'Y': y0})

simulation_data['UE#'] = simulation_data.index.values
simulation_data['Iteration'] = 0
simulation_data['Distance'] = compute_distance(x0, y0)
simulation_data['RSRP_LTE'] = compute_power(d=simulation_data['Distance'], tx=46, g=17, loss=3, f=2100)
simulation_data['RSRP_5G'] = compute_power_5g(d=simulation_data['Distance'], tx=46, g=24, loss=3, f=(f_mmWave / 1000.))
#simulation_data['SINR'] = compute_sinr_lte(simulation_data['RSRP_LTE'], simulation_data['UE#'] )
simulation_data['Gap_Closed'] = (simulation_data['RSRP_LTE'] >= event_a1_measurement_gap)
simulation_data['Gap_Open'] = (simulation_data['RSRP_LTE'] <= event_a2_measurement_gap)

simulation_data['HO_executed'] = (simulation_data['Gap_Open'] & (simulation_data['RSRP_5G'] >= RSRP_5G_min))

average_UE_speed = 1000. * simulation_data['Distance'].mean() / max_sim_time # original unit is meter/ms = km/s.  Multiplied by 1e3 to make m/s.
T_coherence = np.ceil(average_UE_speed / c * f_mmWave) # unit is in LTE subframes

print('Average UE speed = {:0.6f} m/s.'.format(average_UE_speed)) 
print('mmWave channel coherence time is: {0:.0f} LTE subframes.'.format(T_coherence))

N_training = int(min(T_coherence, r_training * max_sim_time))

# Now create a simulation time
for simulation_time in 1 + np.arange(max_sim_time):
    # Now move the users
    simulation_n = pd.DataFrame()
    
    u_1 = np.random.uniform(0.0, 1.0, n) # generate n uniformly distributed points 
    u_2 = np.random.uniform(0.0, 1.0, n) # generate another n uniformly distributed points 
    
    ([x, y]) = plot_network(u_1, u_2)

    # compute the SINR and RSRP of all UEs
    simulation_n = pd.DataFrame({'X': x,
                                'Y': y})
    simulation_n['UE#'] = simulation_n.index.values
    simulation_n['Iteration'] = simulation_time
    simulation_n['Distance'] = compute_distance(x, y)
    simulation_n['RSRP_LTE'] = compute_power(d=simulation_n['Distance'])
    simulation_n['RSRP_5G'] = compute_power_5g(d=simulation_n['Distance'])
    
    # Now based on the received power, handover yes or no?
    simulation_n['Gap_Closed'] = (simulation_n['RSRP_LTE'] >= event_a1_measurement_gap)
    simulation_n['Gap_Open'] = (simulation_n['RSRP_LTE'] <= event_a2_measurement_gap) # this is a handover request (almost)
    
    simulation_n['HO_executed'] = (simulation_n['Gap_Open'] & (simulation_n['RSRP_5G'] >= RSRP_5G_min))
    
    # just append this record
    simulation_data = simulation_data.append(simulation_n, ignore_index=True)

# Now go over a given UE at any time
simulation_result = pd.DataFrame()

for i in np.arange(n):
    UE_i=simulation_data[simulation_data['UE#']==i]
    UE_i_RSRP_LTE=UE_i['RSRP_LTE']
    UE_i_RSRP_5G=UE_i['RSRP_5G']
    UE_i_HO_Executed = UE_i['HO_executed']
    UE_i_Gap_Open = UE_i['Gap_Open']
    # Compute handover success rate
    UE_i_HO_Succ =UE_i_HO_Executed.sum() / UE_i_Gap_Open.sum() # success / attempted by gap open.
    print('For UE #{0} the handover success rate is {1:.2f}%'.format(i, UE_i_HO_Succ * 100))

    # This is where the proposed algorithm takes place.
    UE_i['HO_executed_proposed'] = predict_handover(UE_i_RSRP_LTE, UE_i_RSRP_5G, UE_i['HO_executed'], i) # this line is causing a warning
    UE_i_HO_Executed_Proposed = UE_i['HO_executed_proposed']
    UE_i_HO_Prop_Succ = UE_i_HO_Executed_Proposed.sum() / UE_i_Gap_Open.sum() # success / attempted by gap open.
    print('For UE #{0} the proposed handover success rate is {1:.2f}%'.format(i, UE_i_HO_Prop_Succ * 100))
    
    # TODO: Fix the appended dataframe.
    df = pd.DataFrame({'UE#': i,
                       'Original': [UE_i_HO_Executed.sum()],
                       'Proposed': [UE_i_HO_Executed_Proposed.sum()],
                       'Attempted': [UE_i_Gap_Open.sum()]})
    
    simulation_result= simulation_result.append(df)

    fig = plt.figure(figsize=(7,3))
    plt.title(r'\textbf{UE \#' + '{}'.format(i) + r' Handover Executions}')
    plt.yticks(np.arange(0,1.01, 0.5))
    
    plot_proposed, = plt.step(np.arange(max_sim_time + 1), UE_i['HO_executed_proposed'], c='blue', linestyle='-')
    plot_original, = plt.step(np.arange(max_sim_time + 1), UE_i['HO_executed'], c='black', linestyle=':')
    
    plt.gca().set_yticks([0,1])
    
    p = patches.Rectangle(
        (0, 0), N_training, 1.0,
        alpha=0.2, facecolor="black")
   # plt.plot(np.arange(max_sim_time + 1), UE_i['HO_executed'], c='black')
    plt.legend([plot_proposed, plot_original], ['Proposed', 'Baseline'], loc='lower left')
    plt.gca().add_patch(p)
    
    plt.grid(True)
    plt.xlabel('Time (ms)')
    plt.ylabel('Decision')
    fig.tight_layout()
    plt.savefig('figures/ue{}_decisions.pdf'.format(i), format='pdf')
    plt.show()

    fig = plt.figure(figsize=(7,3))
    plt.title(r'\textbf{UE \#' + '{}'.format(i) + r' Radio Measurements}')
    
# Burn out the first sample, since it represents d0 for the new 5G model
#    plt.plot(1 + np.arange(max_sim_time), UE_i_RSRP_LTE[1:], c='black')
#    plt.plot(1 + np.arange(max_sim_time), UE_i_RSRP_5G[1:], c='gray')
 
    lte_plot, = plt.plot(np.arange(max_sim_time + 1), UE_i_RSRP_LTE, 'k--', label='Sub-6 GHz RSRP')
    mmwave_plot, = plt.plot(np.arange(max_sim_time + 1), UE_i_RSRP_5G, c='gray', label='mmWave RSRP')
 
    plt.grid(True)
    plt.axhline(y=event_a2_measurement_gap, xmin=0, xmax=1, c="red", linewidth=1.5)
    plt.axhline(y=event_a1_measurement_gap, xmin=0, xmax=1, c="blue", linewidth=1.5)
    plt.axhline(y=RSRP_5G_min, xmin=0, xmax=1, c="green", linewidth=1.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('RSRP (dBm)')
    plt.legend(loc=1) # upper right
    fig.tight_layout()
    plt.savefig('figures/ue{}_received_power.pdf'.format(i), format='pdf')
    plt.show()
    
simulation_result = simulation_result.reset_index()
simulation_result = simulation_result.drop(['index', 'UE#'], axis=1)

print('Original HO failures = {:0} out of {:1}'.format(simulation_result['Attempted'].sum() - simulation_result['Original'].sum(), simulation_result['Attempted'].sum()))
print('Proposed HO failures = {:0} out of {:1}'.format(simulation_result['Attempted'].sum() - simulation_result['Proposed'].sum(), simulation_result['Attempted'].sum()))
print('Original HO success rate = {:.6f}%'.format(100*simulation_result['Original'].sum()/simulation_result['Attempted'].sum()))
print('Proposed HO success rate = {:.6f}%'.format(100*simulation_result['Proposed'].sum()/simulation_result['Attempted'].sum()))


######################################################################################################
# This is the true way to plot on the secondary Y axis.
'''
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

plt.title(r'\textbf{UE \#0 Radio Measurements (LTE)}')
ax2.set_ylim(0,2)
ax1.plot(np.arange(max_sim_time + 1), UE_0_RSRP,c='blue')
ax2.plot(np.arange(max_sim_time + 1), UE_0_SINR,c='red')
plt.grid(True)
ax1.axhline(y=event_a2_measurement_gap, xmin=0, xmax=6, c="black", linewidth=1.5)
plt.xlabel('Time')
ax1.set_ylabel('RSRP (dBm)')
ax2.set_ylabel('SINR (dB)')
plt.savefig('figures/ue0_received_power.pdf', format='pdf')        
plt.show()

'''