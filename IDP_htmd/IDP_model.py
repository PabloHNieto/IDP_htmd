from htmd.model import Model 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as colormaps
from matplotlib.lines import Line2D  
from htmd.ui import *
from tqdm import tqdm
import networkx as nx
from htmd.util import ensurelist

class IDPModel(object):
	def __init__(self, model):
		self.Gc = None
		self.model = model
		self.min_rate = 0
		self.modelToNetwork(self.min_rate)

	def modelToNetwork(self, min_rate=0.1):
		""" Creates a network representation the MSM.

		Creates a graph representation of the markov state model using as nodes 
		the microstates and the transition between them as the edges
		Executed at initializing the IDPModel

		Parameters
		----------
		min_rate : float
				A floating point number between 0 and 1 indicating the minimum transition rate
			between two microsates to be included as an edge

		Examples
		--------
		>>> IDPmodel = IDPModel(model)
		>>> IDPmodel.modelToNetwork(min_rate=0.05)
		>>> print(IDP.Gc)
		"""
		self.min_rate = min_rate
		states = self.model.msm.active_set
		Q = self.model.msm.transition_matrix
		count_matrix = self.model.msm.count_matrix_active
		self.Gc = nx.DiGraph()

		for i, originSt in enumerate(states):
			for j, destinationSt in enumerate(states):
				rate = Q[i][j]
				if rate>=min_rate and originSt!=destinationSt and rate>0:
					self.Gc.add_edge(originSt,
										destinationSt, 
										weight= -np.log(rate))

	def findPath(self, source=[], target=[]):
		# all_paths = nx.shortest_path(self.Gc)
		# all_paths = nx.dijkstra_path(self.Gc)
		out = {}
		for s in source:
			tmp_path = {}
			for t in target:
				try:
					tmp_path[t] = nx.dijkstra_path(self.Gc, s, t)
				except:
					pass
			if tmp_path != {}:
				out[s] = tmp_path
		return out

	def plotPathwayGraph(self, paths, pos=None, transition=None, weigthed=False, **kwds):
		""" Creates a network representation the MSM.

		Draw the graphs highlighting the selected paths.

		Parameters
		----------
		paths : array
				Array of paths leading from sources to target nodes to be plotted
		pos : [float]
				Array with the position of the nodes
		transition: boolean
				Array with the data to draw the transition 
		weigthed: boolean
				Whether to draw taking into account the eq. population of the macrostate
		"""

		try:
			ax
		except:
			f, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))

		if not pos:
			pos = self._set_node_position(self.Gc, self.model, weighted=weigthed)
		
		subgraphs = [ self.Gc.subgraph(path) for path in paths ]

		temp_min_rate = self.min_rate
		self.modelToNetwork(0.1) ##Calculating graph with less nodes

		for subgraph in subgraphs:
			self.Gc = nx.compose(self.Gc, subgraph)

		small_pos = self._set_node_position(self.Gc, self.model, weighted=weigthed)

		sub_pos = { node:pos[node] for node in self.Gc.nodes }

		self.plotGraph(ax=ax,
			pos=small_pos,
			node_color="red",
			node_size= 0.5, 
			returnValues=True,
			edge_color='gray', 
			alpha=0.9,
			legend=False,
			arrows=False)

		for path in paths:
			subgraph = self.Gc.subgraph(path)
			full_path = [(path[i], path[i+1]) for i, z in enumerate(path[0:-1])]

			sub_position = { node:pos[node] for node in path }

			node_color = "blue"
			if transition:
				node_color = []
				for i in subgraph.nodes:
					if(i == transition['bulk']):
						node_color.append(transition['bulk_color'])
					else:
						node_color.append(transition['color_by_macro'][self.model.macro_ofmicro[i]])
			
			nx.draw_networkx_nodes(subgraph, 
				pos=sub_position, 
				node_size=300, 
				node_color=node_color, 
				width=3, 
				ax=ax)
			nx.draw_networkx_labels(subgraph, 			
				pos=sub_position,
				font_color="white",
				ax=ax) 
			nx.draw_networkx_edges(subgraph,
				edgelist=full_path, 
				pos=sub_position, 
				width=3, 
				ax=ax)

		self.modelToNetwork(temp_min_rate)

	def plotGraph(self, rate=0.1, pos=None, node_color=None, node_size=None, weigthed=False,
		legend=True, legend_label=None, legend_color=None, label=False,
		top_transition=11, returnValues=False, plot=True, **kwds):
		""" Plots the microstate network of the model

		Creates a graph representation of the markov state model using as nodes 
		the microstates and the transition between them as the edges
		Executed at initializing the IDPModel

		Parameters
		----------
		color : []
			Array of node colors
		position : [] 
			Array with the positions of each node. Calculated by default setting the most populated
			node in the center and the rest with a distance to the center proporcional to their connectivity degree.
		node_color: 
		weigthed : bool
			Whether to divide the graph into section represenative of macro population or not.
		top_transition : int
			A number bigger than one indicating the number of edges accounting with more transition
			in the network to higlight
		"""

		# cc = nx.closeness_centrality(self.Gc, distance='weight')

		if not pos:
			all_pos = self._set_node_position(self.Gc, self.model, weighted=weigthed)
		else:
			all_pos = pos

		# width, edge_color = self._getPlotWidthEdges(self.Gc, self.model, top_transition=top_transition)


		tmp_min_rate = self.min_rate

		self.modelToNetwork(rate)

		if not node_color:
			node_color = self._getPlotColor(self.model, self.Gc)
		
		if not node_size:
			# node_size =  self._getMicroPop(self.model, self.Gc)
			node_size = self.model.msm.stationary_distribution[self.Gc.nodes]
		
		pos = { node: all_pos[node] for node in self.Gc.nodes }
		# [print(cc[i]) for i in self.Gc.nodes]
		if plot:
			
			try:
				ax
			except: 
				plt.figure(figsize=(8,8))

			nx.draw(self.Gc, node_color=node_color, 
				pos=pos, node_size=node_size*30000, 
				# edge_color=edge_color, width=width,
				edge_cmap=plt.cm.Blues, with_labels=label, **kwds)

			if legend:
				custom_lines = [Line2D([0], [0], color=[1, 0, 1], lw=5, ),
												Line2D([0], [0], color=[0, 0, 1], lw=5, ),
												Line2D([0], [0], color=[0, 1, 1], lw=5, )]
				label = ['Beta', 'Random Coil', 'Helix']
				if legend_color:
					custom_lines = [Line2D([0], [0], color=i, lw=5) for i in legend_color]

				if legend_label:
					label = legend_label

				try:
					ax.legend(custom_lines, label, prop={'size': 10}, loc="upper right")
				except: 
					plt.legend(custom_lines, label, prop={'size': 10})

		self.modelToNetwork(tmp_min_rate)
		if returnValues:
			return pos, node_size, node_color

	def _angles_range(self, model):
		angle_value = [pop*360 for macronum, pop in zip(range(model.macronum), model.eqDistribution(plot=False)) ]
		angle_range = []
		for idx, i in enumerate(angle_value):
			if idx == 0:
				angle_range.append([0, i])
			else:
				last_value = angle_range[-1][-1]
				angle_range.append([ last_value, last_value + i ])
		return np.array(angle_range)

	def _set_node_position(self, graph, model, max_radius=5, weighted=False):
		degrees = self.Gc.degree()
		out = np.array([np.array([i[0] , i[1]]) for i in degrees])
		cc = nx.closeness_centrality(self.Gc, distance='weight')

		try:
			max_conection = np.max(out[:,1])
		except:
			print("Empty graph")
			return
			
		position = {}
		for i in graph.nodes:
			#Calculate distance from center 
			if degrees[i] == max_conection:
				position[i] = np.array([0, 0])
				continue
			else:
				# distance_from_center = max_radius - max_radius * degrees[i]/max_conection
				distance_from_center = (1 - cc[i]) * max_radius
			#Select angle 
			macro = model.macro_ofmicro[i]
			if weighted:
				angle_range = self._angles_range(model)
				angle = np.random.randint(angle_range[macro][0] + 2, angle_range[macro][1] - 2)
			else:
				angle = np.random.randint(macro*360/model.macronum + 2, (macro + 1)*360/model.macronum - 2)                
			#Calculate coordinates
			x = distance_from_center * np.sin(np.radians(90 - angle))
			y = distance_from_center * np.sin(np.radians(angle))
			#Set position
			position[i] = np.array([x, y])

		return position

	def _getPlotWidthEdges(self, graph, model, type="both", top_transition=10):
		total_w = []
		for node in graph.nodes:
			for reach_macro in graph[node]:
				if (node != reach_macro):
					weigth = model.msm.count_matrix_active[node][reach_macro] + model.msm.count_matrix_active[reach_macro][node]
					total_w.append(weigth)

		width = []
		edge_color = []
		max_change = max(total_w)
		total_w.sort()
		for edge in graph.edges:
			weigth = model.msm.count_matrix_active[edge[0]][edge[1]] + model.msm.count_matrix_active[edge[1]][edge[0]]
			color = weigth/max_change
			if (np.any(np.array(total_w[0:top_transition*2]) == weigth)):
				#         l_edge_color.append([(1 - color)%1, (1 - color)%1, (1 - color)%1])
				edge_color.append([(1 - color)%1, 0, 0])
				width.append(4)
			else:
				edge_color.append([(1 - color)%1, (1 - color)%1, (1 - color)%1])
				width.append(2)
		return width, edge_color

	def _getPlotColor(self, model, graph):
		node_color = {} 
		for idx, micro in enumerate(self.data_by_micro['ss']):
			ss_type_mean = [np.mean(micro == ss_type) for ss_type in [1, 2]]
			node_color[idx] = ss_type_mean 
		node_color_final = []
		for i in graph.node:
			data = node_color[i]
			tmp_color = [1, 1, 0]
			max_position = np.where(data == np.max(data))[0][0]
			tmp_color[max_position] = 1 - data[max_position]*2
			node_color_final.append(tmp_color)
		return node_color_final

def plot_contacts(data, mol, labels=None, idx=None, mod=None, title="Default Title", plot=True, save=None, vmin=0, vmax=1, cmap="viridis"):
	if not idx:
		idx = mol.resid[0]
	xlabels = [b + str(i+idx) for i,b in enumerate(mol.sequence()['P1'])]
	x = [i-0 for i in range(len(xlabels))]

	from mpl_toolkits.axes_grid1 import make_axes_locatable

	plt.figure(figsize=(8, len(data)), dpi=300)
	ax = plt.gca()
	im = ax.imshow(np.flip(data, 0)[::-1], vmin=vmin, vmax=vmax, cmap=cmap)

	plt.title(title)

	ax.set_xticklabels(xlabels, rotation='vertical', ha="center")
	ax.tick_params(length=0)
	ax.set_xticks(x)

	if not labels:
		labels = ['Macro-{}'.format(i) for i in range(len(data))]

	if mod:
		ylabels = ['Macro {}: {:5.2f}%'.format(idx, i*100) for idx, i in enumerate(mod.eqDistribution(plot=False),0 )]
		y = [i for i in range(mod.macronum)]
		ax.set_yticks(y)
		ax.set_yticklabels(ylabels)
	else:
		y = [i for i in range(len(labels))]
		ax.set_yticks(y)
		ax.set_yticklabels(labels)
	ax.set_xlabel("p27 Sequence")


	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)

	plt.colorbar(im, cax=cax)

	if plot:
		plt.show()

	if save:
		plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)

def get_data(mod, metr, skip=1):
	""" Returns the projected data of metric applied to a model


		Parameters
		----------
		mod : htmd.Model
			Model to get the simlist
		metric : htmd.MetricData
			MetricData with the metric we want to project
		skip : int
			Frames to skip while projecting the data. Default = 1
		"""
	metric = Metric(mod.data.simlist, skip = skip)
	metric.set(metr)
	data = metric.project()
	return data

def plotDataByState(data, states, name="ss", stateType="macro"):
	for macro, col in zip(data, list("rbgy")):
		plt.plot(range(1, 1+len(macro[0])), macro[0], color=col, marker="o")
		if name == "ss":        
			plt.plot(range(1, 1+len(macro[0])), macro[1]*-1, color=col, marker="o")
	plt.show()

def _atom_contact_plot(ver, ax, mol, mapping, label, cmap="viridis"):

	three_letter ={'V':'VAL', 'I':'ILE', 'L':'LEU', 'E':'GLU', 'Q':'GLN', \
		'D':'ASP', 'N':'ASN', 'H':'HSD', 'W':'TRP', 'F':'PHE', 'Y':'TYR', \
		'R':'ARG', 'K':'LYS', 'S':'SER', 'T':'THR', 'M':'MET', 'A':'ALA', \
		'G':'GLY', 'P':'PRO', 'C':'CYS', 'MOL':'MOL'}

	im = ax.imshow(ver, vmin=0, vmax=1, aspect="equal", cmap=cmap)
	all_res = [three_letter[i]+str(b) for i, b in zip(list(mol.sequence()['P1']), list(set(mol.resid))[1:])] 

	ax.tick_params(length=0)
	ax.set_yticks([len(ver)*0.2])
	ax.set_yticklabels([label], rotation="vertical")
	ax.set_xticks([i+0.3 for i in range(len(list(mol.sequence()['P1'])))])
	ax.set_xticklabels(all_res, rotation="vertical", ha='center')
	ax.set_aspect('auto')
	return im

def contact_plot_by_atom(all_data, mol, mapping=None, label=None, chain_id="P1", title=None, plot=True, save=None, **kwds):
	sequence_length = len(mol.sequence()[chain_id])
	dimension = int(len(all_data[0])/sequence_length)
	f, axes = plt.subplots(ncols=1, 
		nrows=len(all_data), 
		figsize=(10, len(all_data)*1.5), 
		sharex=True)
	if title:
		f.suptitle(title)
	f.subplots_adjust(hspace=0)

	if not label:
		label = [ "Macro-{}".format(i) for i in range(len(all_data))]

	if len(all_data) == 1:
			data = np.transpose(all_data[0].reshape(sequence_length, dimension))
			z = _atom_contact_plot(data, axes, mol, mapping, **kwds)
	else:
		for b,(r, data) in enumerate(zip(axes.flat, all_data)):
			data = np.transpose(data.reshape(sequence_length, dimension))
			z = _atom_contact_plot(data, r, mol, mapping, label[b], **kwds)

	cbar_ax = f.add_axes([0.92, 0.15, 0.025, 0.7])
	f.colorbar(z, cax=cbar_ax)

	if plot:
		plt.show()

	if save:
		plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)

def plot_dihedral(data, mol, start_index=1, chain_id="X", plot=True, save=None, title=None, ylabel=None, **kwds):
	""" Plots the standard deviation of dihedral angles by macrostates

	Parameters
	----------
	data : [] 
	start_index : int
			The number of the first residue of the molecule. For renumbering purposes
	chain_id : str
			The chain ID of the molecule to select in order to create labels
	save : str
			Path of the file in which to save the figure
	"""
	
	label = [b + str(i + start_index) for i,b in enumerate(mol.sequence()[chain_id])]
	newLabel = ["{}-{}".format(label[idx], label[idx + 1]) for idx in list(range(len(label)-1))]

	yLabel = ["Cosine", "Sine"]

	fig, axes = plt.subplots(figsize=(6, len(data)*1.6), 
		nrows=len(data), ncols=1, sharex=True, dpi=300)

	if title:
		fig.suptitle(title)

	try:
		len(axes)
	except:
		axes = [axes]

	for idx, (ax, dat)  in enumerate(zip(axes, data)):
		newSTDsin = _merge(dat[0::2])
		newSTDcos = _merge(-1*dat[1::2])
		xticks = [i-0.51 for i in range(0, len(newSTDsin)+1, 2)]
		ax.bar(range(len(newSTDsin)), newSTDsin, lw=0.5)
		ax.bar(range(len(newSTDsin)), newSTDcos, lw=0.5)

		ax.axhline(y=0.50, xmin=0, xmax=len(dat), linestyle="dashed", color="red", lw=0.5)
		ax.axhline(y=0.15, xmin=0, xmax=len(dat), linestyle="dashed", color="red", lw=0.5)
		ax.axhline(y=0.0, xmin=0, xmax=len(dat), color="black", lw=0.8)
		ax.axhline(y=-0.50, xmin=0, xmax=len(dat), linestyle="dashed", color="red", lw=0.5)
		ax.axhline(y=-0.15, xmin=0, xmax=len(dat), linestyle="dashed", color="red", lw=0.5)
		ax.set_xlim(-2, len(newSTDsin)+2)
		ax.set_ylim(-1.2, 1.2)

		ax.set_xticks(xticks)
		ax.set_yticks([-0.1, 0.7])
		ax.set_xticklabels(newLabel, rotation='vertical', ha="left", fontdict={'size': "small"})
		ax.set_yticklabels(yLabel, rotation='vertical', ha="center")
		if ylabel:
			ax.set_ylabel(ylabel[idx])

		fig.subplots_adjust(hspace=0.0)
	if save:
		plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)

	if plot:
		plt.show()

def _merge(arr):
	half1 = arr[0::2]
	half2 = arr[1::2]
	newArr = [ x for i in zip(half1[1:], half2[:-1]) for x in i ]
	return [half1[0]] + newArr + [half2[-1]]

def _contactVecToMatrix(vector, atomIndexes):
		from copy import deepcopy
		# Calculating the unique atom groups in the mapping
		uqAtomGroups = []
		atomIndexes = deepcopy(list(atomIndexes))
		for ax in atomIndexes:
				ax[0] = ensurelist(ax[0])
				ax[1] = ensurelist(ax[1])
				if ax[0] not in uqAtomGroups:
						uqAtomGroups.append(ax[0])
				if ax[1] not in uqAtomGroups:
						uqAtomGroups.append(ax[1])
		uqAtomGroups.sort(key=lambda x: x[0])  # Sort by first atom in each atom list
		num = len(uqAtomGroups)

		matrix = np.zeros((num, num), dtype=vector.dtype)
		mapping = np.ones((num, num), dtype=int) * -1
		for i in range(len(vector)):
				row = uqAtomGroups.index(atomIndexes[i][0])
				col = uqAtomGroups.index(atomIndexes[i][1])
				matrix[row, col] = vector[i]
				matrix[col, row] = vector[i]
				mapping[row, col] = i
				mapping[col, row] = i
		return matrix, mapping, uqAtomGroups

def _contact_plot(ver, axes,  mol, mapping, ligand=False, title=None, xlabels=None, ylabels=None, legend=False):
		from mpl_toolkits.axes_grid1 import make_axes_locatable
		global mpbl
		vector = np.ones(ver.shape, dtype=bool)
		colors= ver
		truecontacts = np.zeros(len(vector), dtype=bool)
		
		# Creating the 2D contact maps
		cm, newmapping, uqAtomGroups = _contactVecToMatrix(vector, mapping.atomIndexes)
		num = len(uqAtomGroups)
		cmtrue, _, _ = _contactVecToMatrix(truecontacts, mapping.atomIndexes)
		# im = axes.imshow(cmtrue/2,
		im = axes.imshow(cmtrue/2,
										 interpolation='none',
										 vmin=0, vmax=1, 
										 aspect='equal',
										 cmap='gray')
										 # /2 to convert to gray from black
		
		rows, cols = np.where(cm)
		 
		if isinstance(colors, np.ndarray) and isinstance(colors[0], float):
				mpbl = colormaps.ScalarMappable(cmap=colormaps.jet)
				mpbl.set_array(colors)
				colors = mpbl.to_rgba(colors)
		if len(colors) == len(vector):
						colors = colors[newmapping[rows, cols]]
		
		rows = rows + 0.5
		cols = cols + 0.5

		axes.set_xticklabels(xlabels, rotation="vertical", ha='left')
		axes.set_yticklabels(ylabels, va='bottom')
		axes.set_xticks(np.arange(0, num, 1))
		axes.set_yticks(np.arange(0, num, 1))

		axes.scatter(rows, cols,s =100,  c=colors, lw=0, marker="s")
		axes.set_axisbelow(True)
		axes.set_title(title)
		axes.grid(which="both", color='#969696', linestyle='-', linewidth=1)
		axes.tick_params(axis='both', which='both', length=0)
		axes.set_xlim([0, num ])
		axes.set_ylim([0, num ])

		if legend:
			divider = make_axes_locatable(axes)
			cax = divider.append_axes("right", size="5%", pad=0.05)
			plt.colorbar(im, cax=cax)

def contact_plot(data, mol, cols, rows, model=None, plot=True, save=None, **kwargs):
	f, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(cols*10, rows*10),
		dpi=200)

	for dat, ax in zip(data, axes.flat):
			_contact_plot(ver=np.array(dat), axes=ax, mol=mol, **kwargs)

	if plot:
		plt.show()

	if save:
		plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)

def generate_labels(mol, *args):
    three_letter ={'V':'VAL', 'I':'ILE', 'L':'LEU', 'E':'GLU', 'Q':'GLN', \
				'D':'ASP', 'N':'ASN', 'H':'HSD', 'W':'TRP', 'F':'PHE', 'Y':'TYR', \
				'R':'ARG', 'K':'LYS', 'S':'SER', 'T':'THR', 'M':'MET', 'A':'ALA', \
				'G':'GLY', 'P':'PRO', 'C':'CYS', 'MOL':'MOL'}

    all_res = [three_letter[i]+str(b) for i, b in zip(list(mol.sequence()['P1']), list(set(mol.resid[mol.atomselect('protein')])))]

    _ = [ all_res.append(i.upper()) for i in args ]
    return all_res

def plot_mfpt(data, mol, save=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # plt.figure( dpi=300)
    plt.title("MFPT between macros")
    ax2 = plt.gca()
    im = plt.imshow(np.log10(data), cmap='tab20c')
    # ax2.colorbar()
    #
    # _ = ax2.set_xticks(range(len(sorted_model_labels)))
    # _ = ax2.set_xticklabels(sorted_model_labels2, rotation='vertical')
    #
    # _ = ax2.set_yticks(range(len(sorted_model_labels)))
    # _ = ax2.set_yticklabels(sorted_model_labels2)
    ax2.set_ylabel("From ...")
    ax2.set_xlabel("To ...")
    #
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    cbar = plt.colorbar(im, cax=cax)
    # $\it{Italics}$
    cbar.set_label("$\it{log_{10} (mfpt)}$")
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)

if __name__ == "__main__":
	pass