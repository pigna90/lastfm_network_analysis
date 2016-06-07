from Demon import Demon
import numpy as np
import matplotlib.pyplot as plt

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
	pos = np.arange(len(x))
	width = 0.5     # gives histogram aspect to the bar diagram
	
	ax = plt.axes()
	ax.set_xticks(pos + (width / 2))
	ax.set_xticklabels(x)
	plt.xticks(rotation="vertical")
	plt.bar(pos, freq, width, color='g',alpha=0.6,linewidth=0)
	plt.margins(0.01)

	if (xlabel != None and ylabel != None):
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		
	if out == None:
		plt.show()
	else:
		plt.savefig(out,bbox_inches="tight")

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

	histogram(x,freq,"Epsilon","Number of communities")
	
def main():
	demon_analysis("../data/network_cleaned16.csv",(0.001,0.4),3,2,"/tmp/demon")

if __name__ == "__main__":
	main()
