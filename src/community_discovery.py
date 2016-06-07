from Demon import Demon
import numpy as np
import matplotlib.pyplot as plt
import os

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
	plt.xticks(range(len(fake_label)),fake_label, size='small',rotation='horizontal')
	
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

	#histogram(x,freq,"Epsilon","Number of communities","/tmp/demon")

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
def plot_epsilon_dict():
	l = {}
	dict_list = os.listdir("../demon_log/")
	dict_list.sort()
	for d in dict_list:
		l[float(d.split("_")[2])] = len(dict_from_file("../demon_log/" +d))
	x = []
	freq = []
	for i in sorted(l):
		x.append(i)
		freq.append(l[i])
	histogram(x,freq,"Epsilon","Number of communities","/tmp/demon")

def main():
	demon_analysis("../data/network_cleaned16.csv",(0.001,0.4),3,60,"/tmp/demon")

if __name__ == "__main__":
	main()
