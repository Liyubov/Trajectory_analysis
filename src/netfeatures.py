import numpy as np
#from scipy.spatial import ConvexHull
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


'''
Here we estimate network measures from a given graph, encoded as networkx object, e.g. 
network = nx.watts_strogatz_graph(n, k, p)
'''

def degree(network): 
	deg = nx.degree_centrality(network) # gives a dictionary  	#print(type(G.degree))# gives 'networkx.classes.reportviews.DegreeView'
	degree_sequence = sorted([d for n, d in network.degree()], reverse=True) # gives array of degree values
	return degree_sequence

def betweenness(network):
	betw = nx.betweenness_centrality(network) # gives betweenness  
	bet_sequence = list(nx.betweenness_centrality(network).values()) # gives array of betweenness values
	return bet_sequence
	
	
def clustering(network): 
	clust = nx.clustering(network) # gives clustering of nodes  
	clust_sequence = list(nx.clustering(network).values()) # gives array of clustering values
	return clust_sequence
	
	
def pad(feature):
    max_value = np.max([len(i) for i in feature])
    return np.array([np.pad(i,(0,max_value-len(i))) for i in feature])

def cv(t, i, delta):
    try:
        hul = ConvexHull(t[i:i+delta,:])
        return hul.volume
    except:
        return 0

def convex_hull(trajectories, delta, padding=True):
    '''
    This function calculates the value of the convex hull as function of the length of the trajectory. 
	Note that we define a seperate function 
    that ensures the scipy ConvexHull function doesn't return an error when the value is 0
    '''
    traj = [ i.T for i in trajectories]
    vol_tot = [np.array([cv(traject,i,delta) for i in np.arange(len(traject)-delta-1)]) for traject in traj]
    if padding == True:
        return pad(vol_tot)
    else:
        return vol_tot
    
def Q_measure(trajectories, window_length, padding=True):
    '''
    This function calculates the Q measure hull measure for an ensemble of trajctories. The input has to be a list of x and y coordinates arrays as well as the \tau value. This function returns the convex hull time-series and the length of each series  
    '''
    Q = [np.sum((trajectory[:,:-window_length] + trajectory[:,window_length:])**2, axis=0) for trajectory in trajectories]
    Q = [k / (np.max(k)+10e-6) for k in Q]
    if padding == True:
        return pad(Q)
    else:
        return Q 
    
def dist_distribution(trajectories):
    '''
    This function calculates the distribution of distance travelled per unit of time and retuns an histogram of this distribution. Note that we fixed the range and number of bins for simplicity, but this might need to be 
    '''
    step_distance = [np.sqrt(np.diff(trajectory, axis=0)**2) for trajectory in trajectories]
    distributions = np.array([np.array(np.histogram(dist, bins=20)[0]) for dist in step_distance])
    return distributions
	
'''
def traject_itself(trajectories,padding=True): 
	
	#x and y coordinates array of trajectory put together
	
	#print(np.shape(trajectory))
	#trajectory_coordinates = [np.concatenate((trajectory[0,:], trajectory[1,:]), axis=0).reshape(trajectory[1,:].size*2) for trajectory in trajectories] 

	if (padding == True):
		return pad(trajectory_coordinates)
    else:
		return trajectory_coordinates
	return trajectory_coordinates 

#	for trajectory in trajectories: 
#		print( np.concatenate((trajectory[0,:], trajectory[1,:]), axis=1))
'''
	
