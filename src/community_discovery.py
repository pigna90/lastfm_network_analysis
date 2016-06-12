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
		c = list(map(list,c))
		out = open(out_path + str(k) + "_clique.dat","w")
		for community in c:
			out.write("%d\t[" % c.index(community))
			for node in community:
				out.write('"%s",' % node)
			out.write("]\n")
		out.close()

##
# Deserialize DEMON results from file and return a list of first n
# ordered communities for every file read
##
# Params:
# eps_list - List of epsilon to read
# log_demon - Path to demon log folder
# n - number of communities for each eps
##
def deserialize_demon_results(eps_list,log_demon,n):
	list_communities = []
	for eps in eps_list:
		dict_list = os.listdir(log_demon)
		for d in dict_list:
			if eps in d:
				list_dict_values = list(dict_from_file(log_demon+d).values())
				list_dict_values.sort(key=len,reverse=True)
				list_communities.append(list_dict_values[:n])
	return list_communities

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
# Load dicts from file made by DEMON with different epsilon and
# plot communities frequencies
##
def plot_epsilon_dict(log_directory="../demon_log/",out=None):
	l = {}
	dict_list = os.listdir(log_directory)
	dict_list.sort()
	for d in dict_list:
		l[float(d.split("_")[2])] = len(dict_from_file(log_directory +d))
	x = []
	freq = []
	for i in sorted(l):
		x.append(i)
		freq.append(l[i])
	histogram(x,freq,"Epsilon","Number of communities",out)

##
# Plot jaccard heatmap calculated on comunity result serialized on file.
##
# Params:
# communities - list of community
# out - Path for output plot result
## 
def plot_jaccard_heatmap(communities,out=None):
	data =np.array(list(map(jaccard_similarity,list(product(communities, repeat=2)))))
	data = data.reshape(30,30)
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
	print(counts)
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

def main():
	graph = "../data/network_cleaned16.csv"
	external_data = "../data/artists_genres.csv"
	log_demon = "../demon_log/"
	eps_list = list(map(str,[0.034,0.234,0.301,0.368]))

	# Make DEMON analysis on eps range and serialize results on file
	#demon_analysis(graph,(0.001,0.4),3,60,"/tmp/demon")
	#plot_epsilon_dict(out="/tmp/demon")

	# Loading graph from file
	G=read_graph(graph)

	# Reading ommunities serialized by DEMON
	#list_communities = deserialize_demon_results(eps_list,log_demon,30)

	# Legend for plot distribution
	#legend = ["eps = " + eps for eps in eps_list]
	
	#plot_distribution(distribution_type="Nodes",list_communities=list_communities,legend=legend,graph=G)
	#plot_distribution(distribution_type="Density",list_communities=list_communities,legend=legend,graph=G)
	#plot_distribution(distribution_type="Transitivity",list_communities=list_communities,legend=legend,graph=G)
	
	#plot_jaccard_heatmap(list_communities[3])

	#for dim in ["artist","genre"]:
		#for c in [7,8,9]:
			#out = "/tmp/" + str(c) + "_" + dim
			#data = list_communities[2]
			#community = data[c]
			#plot_pie_external(data=external_data,dim=dim,pie_pieces=8,community=community)

	## Plot graph with communities			
	#data = list_communities[2]
	#data = [data[7],data[8],data[9]]
	#colors = ['yellowgreen', 'gold', 'lightskyblue']
	#plot_communities(G,data,colors)

	#k_list = list(range(2,10))
	#k_clique_analysis(G,k_list,"../data/k-clique/")
	
	k_clique_communities = deserialize_demon_results(list(map(str,[4])),"../data/k-clique/",10)

	legend = ["k = " + str(k) for k in [3,4,5,6]]
	
	#plot_distribution(distribution_type="Nodes",list_communities=k_clique_communities,legend=legend,graph=G,out="/tmp/k-clique_nodes")
	#plot_distribution(distribution_type="Density",list_communities=k_clique_communities,legend=legend,graph=G,out="/tmp/k-clique_density")
	#plot_distribution(distribution_type="Transitivity",list_communities=k_clique_communities,legend=legend,graph=G,out="/tmp/k-clique_transitivity")

	#colors = ['yellowgreen', 'gold', 'lightskyblue','royalblue','magenta','r']
	#for data in k_clique_communities:
		#plot_communities(G,data[:5],colors,out=str(k_clique_communities.index(data)+3)+"_clique_graph")

	#c_4 = (k_clique_communities[0])[:3]
	#for community in c_4:
		#plot_pie_external(external_data,"artist",community)
		#plot_pie_external(external_data,"genre",community)
		#quit()

if __name__ == "__main__":
	main()
