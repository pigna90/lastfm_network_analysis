from Demon import Demon
import numpy as np
import matplotlib.pyplot as plt
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
	fake_label = list(map(str,x))
	for i in range(0,len(fake_label)-1):
		if (i%5 != 0):
			fake_label[i] = ""

	plt.bar(range(len(freq)),freq,color='g',alpha=0.6,linewidth=0)
	plt.xticks(range(len(fake_label)),fake_label, size='small',rotation='vertical')
	
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
# Plot two type of distribution analysis computed on a set of comunity.
##
# Params:
# distribution_type - Type of distribution analysis {density,transitivity,nodes}
# graph - Main network contain communities for analysis
# eps_list - List of epsilon to analyze
# log_directory - Directory of comunity analysis results
# out - Path for output plot result
##
def plot_distribution(distribution_type,graph,eps_list,log_directory="../demon_log/",out=None):
	x = [i for i in range(0,30)]
	legend = []
	eps_list = list(map(str,eps_list))
	for eps in eps_list:
		dict_list = os.listdir(log_directory)
		for d in dict_list:
			if eps in d:
				list_dict_values = list(dict_from_file(log_directory+d).values())
				list_dict_values.sort(key=len,reverse=True)
				if distribution_type.lower() == "nodes":
					y = list(map(len,list_dict_values[:30]))
				else:
					y = []
					for l in list_dict_values[:30]:
						H = graph.subgraph(l)
						if distribution_type.lower() == "density":
							y.append(nx.density(H))
						elif distribution_type.lower() == "transitivity":
							y.append(nx.transitivity(H))
						else:
							return None
				plt.plot(x,y,linewidth=2,alpha=0.8)
				legend.append("eps = " + eps)

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
# eps - value of epsilon to analyze
# log_directory - Directory of comunity analysis results
# out - Path for output plot result
## 
def plot_jaccard_heatmap(eps,log_directory="../demon_log/",out=None):
	eps = str(eps)
	dict_list = os.listdir(log_directory)
	for d in dict_list:
		if eps in d:
			list_dict_values = list(dict_from_file(log_directory+d).values())
			list_dict_values.sort(key=len,reverse=True)
			data =np.array(list(map(jaccard_similarity,list(product(list_dict_values[:30], repeat=2)))))
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
# eps - value of epsilon to analyze
# data - csv of external data
# community - id of community to analyze
# log_directory - Directory of comunity analysis results
# pie_pieces - number of segment
# out - Path for output plot result
def plot_pie_external(eps,data,dim,community,log_directory="../demon_log/",pie_pieces=10,out=None):
	eps = str(eps)
	dict_list = os.listdir(log_directory)
	for d in dict_list:
		if eps in d:
			list_dict_values = list(dict_from_file(log_directory+d).values())
			list_dict_values.sort(key=len,reverse=True)
			df = pd.read_csv(data)
			counts = df[df["username"].isin(list(set(list_dict_values[community])))][dim].value_counts()
			counts = counts[:pie_pieces]
			other = pd.Series([abs(sum(counts[:pie_pieces])-sum(counts[pie_pieces:]))],index=["Other"])
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

def plot_demon_communities(graph,communities,out=None):
	nodes = [y for x in communities for y in x]
	nodes = list(set(nodes))
	colors = ['yellowgreen', 'gold', 'lightskyblue', 'white']
	class_colors = {}
	for n in nodes:
		if n in communities[0] and n in communities[1] and n in communities[2]:
			class_colors[n] = colors[3]
		elif n in communities[1]:
			class_colors[n] = colors[0]
		elif n in communities[2]:
			class_colors[n] = colors[1]
		else:
			class_colors[n] = colors[2]
				
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
	#demon_analysis("../data/network_cleaned16.csv",(0.001,0.4),3,60,"/tmp/demon")
	#plot_epsilon_dict(out="/tmp/demon")

	G=read_graph("../data/network_cleaned16.csv")
	#plot_distribution(distribution_type="Nodes",eps_list=[0.034,0.234,0.301,0.368],graph=G,out="/tmp/nodes_DEMON")
	#plot_distribution(distribution_type="Density",eps_list=[0.034,0.234,0.301,0.368],graph=G,out="/tmp/density_DEMON")
	#plot_distribution(distribution_type="Transitivity",eps_list=[0.034,0.234,0.301,0.368],graph=G,out="/tmp/transitivity_DEMON")

	#plot_jaccard_heatmap(0.301,out="/tmp/jaccard")

	#for dim in ["artist","genre"]:
		#for community in [7,8,9]:
			#out = "/tmp/" + str(community) + "_" + dim
			#plot_pie_external(0.301,data="../data/artists_genres.csv",dim=dim,pie_pieces=8,community=community)
	eps = "0.301"
	log_directory = "../demon_log/"
	dict_list = os.listdir(log_directory)
	list_dict_values = []
	for d in dict_list:
		if eps in d:
			list_dict_values = list(dict_from_file(log_directory+d).values())
			list_dict_values.sort(key=len,reverse=True)
			data = [list_dict_values[7],list_dict_values[8],list_dict_values[9]]
	plot_demon_communities(G,data,"/tmp/DEMON_degree_7-8-9")
			
if __name__ == "__main__":
	main()
