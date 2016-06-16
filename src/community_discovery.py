from Demon import Demon
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import networkx as nx
import os
from itertools import product
import seaborn as sns
import pandas as pd
import community
from sklearn.preprocessing import normalize

##
# Print a bar histogram
##
# Params:
# x - list of labels
# freq - label's frequencies
# xlabel - name for x axis
# ylabel - name for y axis
# out - output name file for figure
##
def histogram(x,freq,xlabel=None,ylabel=None,out=None):
	for i in range(0,len(x)-1):
		if (i%5 != 0):
			x[i] = ""

	plt.bar(range(len(freq)),freq,color='g',alpha=0.6,linewidth=0)
	plt.xticks(range(len(x)),x, size='small',rotation='vertical')
	
	if (xlabel != None and ylabel != None):
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		
	if out == None:
		plt.show()
	else:
		plt.savefig(out+".svg",bbox_inches="tight")

##
# Make a communities analysis using DEMON algorithm written
# by Giulio Rossetti (https://github.com/GiulioRossetti/DEMON).
# The analysis is made by iterating on a range of epsilon values.
##
# Params:
# network - file network as edges list
# epsilon_range - tuple of two values, rappresent epsilon range
# min_community - minimum numbers of element needed to create a community
# bins - distance betwen every epsilon inside the range
# out - output path for result of communities analysis
##
def demon_analysis(network,epsilon_range,min_community,bins,out):
	epsilon = epsilon_range[0]
	x = []
	freq = []
	for i in range(0,bins):
		out_path = out + "_" + str(i) + "_" + str(round(epsilon,3)) + "_" + str(min_community)
		dm = Demon(network,epsilon,min_community,out_path)
		communities = dm.execute()
		freq.append(len(communities))
		x.append(round(epsilon,3))
		epsilon += epsilon_range[1]/bins

##
# Make a community analusis using k-clique algorithm from networkx.
# The analysis is made by iterating on a range of k values.
##
# Params:
# graph - networkx graph
# k_lists - lists of k as integer
# out_path - output path for results of communities analysis
##
def k_clique_analysis(G,k_list,out_path):
	for k in k_list:
		c = list(nx.k_clique_communities(G, k))
		print(type(c[0]))
		quit()
		c = list(map(list,c))
		out = open(out_path + str(k) + "_clique.dat","w")
		for community in c:
			out.write("%d\t[" % c.index(community))
			for node in community:
				out.write('"%s",' % node)
			out.write("]\n")
		out.close()

##
# Make community analysis using Louvain algorithm from communities module,
# and serialize result on file as list of nodes for each communities.
##
# Params:
# graph - networkx graph
# out_path - output path for results of communities analysis
##
def louvain_analysis(graph,out_path):
	partition = community.best_partition(graph)
	out = open(out_path + "louvain_communities.dat","w")
	for c in set(partition.values()):
		out.write("%d\t[" % c)
		comm = [k for k, v in partition.items() if v == c]
		for node in comm:
			out.write('"%s",' % node)
		out.write("]\n")
	out.close()

##
# Deserialize DEMON/K-Clique results from file and return a list of
# first n ordered communities for every file read
##
# Params:
# param - List of epsilon/k to read
# log_path - Path to demon log folder
# n - number of communities for each eps/k
##
def deserialize_demon_kclique(log_path,param=None,n=None):
	list_communities = []
	if param == None:
		dict_list = os.listdir(log_path)
		for d in dict_list:
			list_dict_values = list(dict_from_file(log_path+d).values())
			list_dict_values.sort(key=len,reverse=True)
			if n == None:
				list_communities.append(list_dict_values)
			else:
				list_communities.append(list_dict_values[:n])
	else:
		for p in param:
			dict_list = os.listdir(log_path)
			for d in dict_list:
				if p in d:
					list_dict_values = list(dict_from_file(log_path+d).values())
					list_dict_values.sort(key=len,reverse=True)
					if n == None:
						list_communities.append(list_dict_values)
					else:
						list_communities.append(list_dict_values[:n])
	return list_communities

##
# Deserialize Louvain results from file and return a list of
# first n ordered communities
##
# Params:
# log_path - Path to demon log folder
# n - number of communities to read
##
def deserialize_louvain(log_path,n=None):
	list_dict_values = list(dict_from_file(log_path).values())
	list_dict_values.sort(key=len,reverse=True)
	if n==None:
		return list_dict_values
	else:
		return list_dict_values[:n]

##
# Plot two type of distribution analysis computed on a set of comunity.
##
# Params:
# distribution_type - Type of distribution analysis {density,transitivity,nodes}
# legend - plot legend
# graph - Main network contain communities for analysis
# list_communities - lists of communities
# out - Path for output plot result
##
def plot_distribution(distribution_type,legend,graph,list_communities,out=None):
	x = [i for i in range(0,len(list_communities[0]))]
	for communities in list_communities:
		if distribution_type.lower() == "nodes":
			y = list(map(len,communities))
		else:
			y = []
			for l in communities:
				H = graph.subgraph(l)
				if distribution_type.lower() == "density":
					y.append(nx.density(H))
				elif distribution_type.lower() == "transitivity":
					y.append(nx.transitivity(H))
				else:
					return None
		plt.plot(x,y,linewidth=2,alpha=0.8)
		#plt.yscale("log")

	plt.legend(legend, loc='upper left')
	plt.xlabel("Comunity ID")
	plt.ylabel(distribution_type)

	if out == None:
		plt.show()
	else:
		plt.savefig(out+".svg",bbox_inches="tight")
	plt.close()

##
# Load dict from file. Format:
# key\t["a","b","c"]
##
def dict_from_file(path_dict):
	out_dict = {}
	with open(path_dict, 'r') as f:
		for line in f:
			out_dict[int(line.split("\t")[0])] = eval(line.split("\t")[1])
	return out_dict

##
# Plot histogram of DEMON results with different eps serialized on file 
##
# Params:
# communities_lists - list of results with differents eps
# out - Path for output plot result
##
def histogram_epsilon_frequencies(communities_lists,out=None):
	freq = [len(x) for x in communities_lists]
	freq.sort()
	x = list(range(0,len(freq)))
	histogram(x,freq,"Epsilon","Number of communities",out)

##
# Plot jaccard heatmap calculated on comunity result serialized on file.
##
# Params:
# communities - list of community
# out - Path for output plot result
## 
def plot_jaccard_heatmap(communities,row=30,col=30,out=None):
	data =np.array(list(map(jaccard_similarity,list(product(communities, repeat=2)))))
	data = data.reshape(row,col)
	ax = plt.axes()
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	heat = sns.heatmap(data,cmap=plt.cm.Reds,square=True,linewidths=.5, cbar_kws={"shrink": .5},ax = ax)
	heat.invert_yaxis()
	plt.ylabel("Comunity ID")
	plt.xlabel("Comunity ID")
	plt.yticks(size='small',rotation='horizontal')
	plt.xticks(size='small',rotation='vertical')
	if out == None:
		plt.show()
	else:
		plt.savefig(out+".svg",bbox_inches="tight")
	plt.close()

##
# Plot piechart of communities external data.
##
# Params:
# data - csv of external data
# community - nodes of community
# pie_pieces - number of segment
# out - Path for output plot result
##
def plot_pie_external(data,dim,community,pie_pieces=10,out=None):
	df = pd.read_csv(data)
	counts = df[df["username"].isin(list(set(community)))][dim].value_counts()
	other = pd.Series([abs(sum(counts[:pie_pieces])-sum(counts[pie_pieces:]))],index=["Other"])
	counts = counts[:pie_pieces]
	counts = counts.append(other)
	labels = [i[0] for i in counts.iteritems()]
	colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','violet','tomato','cyan','blueviolet','palevioletred','darkorange','grey']
	colors[pie_pieces] = "grey"
	plt.pie(counts,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True, startangle=90,center=(0, 0))
	if out == None:
		plt.show()
	else:
		plt.savefig(out+".svg",bbox_inches="tight")
	plt.close()

##
# Plot graph with different color for every set of communities
##
# Params:
# graph - networkx graph to plot
# communities - list communities (each communities is a list of nodes)
# colors - colormap
# out - Path for output plot result
##
def plot_communities(graph,communities,colors,out=None):
	nodes = [y for x in communities for y in x]
	nodes = list(set(nodes))
	class_colors = {}
	for n in nodes:
		for c in communities:
			if n in c:
				class_colors[n] = colors[communities.index(c)]
			if(all(n in c for c in communities)):
				class_colors[n] = 'white'

	H = graph.subgraph(nodes)
	d = nx.degree(H)
	nx.draw(H,node_list = list(class_colors.keys()), node_color=list(class_colors.values()),node_size = [v * 5 for v in d.values()],width=0.2)
	if out == None:
		plt.show()
	else:
		plt.savefig(out+".svg",bbox_inches="tight")
	plt.close()

##
# Jaccard similarity between two list.
# (Made for use with map()) 
##
# Params:
# pair - tuple of list
##
def jaccard_similarity(pair):
	x = pair[0]
	y = pair[1]
	intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
	union_cardinality = len(set.union(*[set(x), set(y)]))
	return intersection_cardinality/float(union_cardinality)

##
# Read edges list from file
##
def read_graph(filename):
	# Read graph from edgelist file
	g = nx.Graph()
	f = open(filename)
	for l in f:
		l = l.rstrip().replace(" ", ";").replace(",", ";").replace("\t", ";").split(";")
		g.add_edge(l[0], l[1])
	return g

def example_usage():
	graph = "../data/network_cleaned16.csv"
	external_data = "../data/artists_genres.csv"
	log_demon = "../data/demon_log/"
	log_kclique = "../data/k-clique"
	log_louvain = "../data/louvain"
	eps_list = ["0.034","0.234","0.301","0.368"]
	colors = ['yellowgreen', 'gold', 'lightskyblue','royalblue','magenta','r']

	## Loading graph from file
	G=read_graph(graph)
	
	# Make DEMON analysis on eps range and serialize results on file
	demon_analysis(graph,(0.001,0.4),3,60,"/tmp/demon")

	# Reading ommunities serialized by DEMON
	list_communities = deserialize_demon_kclique(log_demon)
	# Reading eps_list results serialized by DEMON. For each eps
	 only the first 30 communities
	list_communities = deserialize_demon_kclique(log_demon,eps_list,30)

	# Plot histogram of communities made with different eps
	histogram_epsilon_frequencies(list_communities)

	# Legend for plot distribution
	legend = ["eps = " + eps for eps in eps_list]

	# Plot number of nodes/density and transitivity
	# for different eps results
	for d_type in ["Nodes","Density","Transitivity"]:
		plot_distribution(distribution_type=d_type,list_communities=list_communities,legend=legend,graph=G)

	# Jaccard heatmap for communities overlapping on one of eps results
	# communities. For example communities calculated with eps = 0.031
	plot_jaccard_heatmap(list_communities[3])

	 Validation with external data.
	 Select an eps results for analysis. For example eps = 0.031
	data = list_communities[2]
	# For some dimensions of external data...
	for dim in ["artist","genre"]:
		# for some communities...
		for c in [7,8,9]:
			community = data[c]
			plot_pie_external(data=external_data,dim=dim,pie_pieces=8,community=community)

	# Plot graph with communities		
	data = list_communities[2]
	data = [data[7],data[8],data[9]]
	plot_communities(G,data,colors)

	# K-Clique analysis
	k_list = list(range(2,10))
	k_clique_analysis(G,[4],"../data/k-clique/")

	# Deserialize results make with k-clique
	k_clique_communities = deserialize_demon_results(list(map(str,[4])),"../data/k-clique/",10)

	
	legend = ["k = " + str(k) for k in [3,4,5,6]]
	for d_type in ["Nodes","Density","Transitivity"]:
		plot_distribution(distribution_type=d_type,list_communities=k_clique_communities,legend=legend,graph=G)

	# Plot graph with communities	
	for data in k_clique_communities:
		plot_communities(G,data[:5],colors,out=str(k_clique_communities.index(data)+3)+"_clique_graph")

	# Validation with external data
	c_4 = (k_clique_communities[0])[:3]
	for community in c_4:
		plot_pie_external(external_data,"artist",community)
		plot_pie_external(external_data,"genre",community)

	# Louvain analysis
	louvain_analysis(G,log_louvain)
	louvain = [deserialize_louvain("../data/louvain_communities.dat")]

	for d_type in ["Nodes","Density","Transitivity"]:
		plot_distribution(distribution_type=d_type,list_communities=louvain,legend=["louvain"],graph=G)

	plot_communities(G,(louvain[0])[:5],colors)

	c_3 = (louvain[0])[:3]
	for community in c_3:
		plot_pie_external(external_data,"artist",community,out=str(c_3.index(community))+"_artist")
		plot_pie_external(external_data,"genre",community,out=str(c_3.index(community))+"_genre")
	
if __name__ == "__main__":
	example_usage()
