"""Class to conver a FluxController object to a graph
"""
import numpy as np
import networkx as nx
from IDP_htmd.FluxController import FluxController 


class Graph_MSM():
    def __init__(self, flux):
        self.graph = None
        self.flux = flux
        self.model_to_network()


    def model_to_network(self):
        """Summary
        """
        self.graph = nx.DiGraph()
        # self.graph = nx.Graph()

        #Initilize nodes
        for i in self.flux.newsets:
            self.graph.add_node(i, flux=0)
            # self.graph.node[i]["flux"] = 0
            

        #Adding weights and connections
        percent_fluxes = 100 * self.flux.pathfluxes / np.sum(self.flux.pathfluxes)
        for path, flux in zip(self.flux.paths, percent_fluxes):
            for idx, node in enumerate(path[0:-1]):
                start = self.flux.newsets[node]
                end = self.flux.newsets[path[idx + 1]]
                
                #Updating nodes weight
                self.graph.node[end]["flux"] += flux
                

                #Updating edges weight
                if self.graph.has_edge(start, end):
                    self.graph[start][end]['weight'] += flux
                else:
                    self.graph.add_edge(start, end, weight=flux)

def hierarchical_layout(msm_graph, source_pos=(0, 2), sink_pos=(0, -2), source=None, sink=None):
    import networkx as nx

    x_step = 1
    
    graph = msm_graph.graph
    degrees = np.array(graph.in_degree(weight="weight"))
    degrees = degrees[np.argsort(degrees[:,1])]
    if not source:
        source = degrees[np.argmin(degrees[:,1])][0]
    if not sink:
        sink = degrees[np.argmax(degrees[:,1])][0]
    
    sorted_nodes = degrees[np.where((degrees[:,1]>0) | (degrees[:,0]==source)), 0][0]
    sorted_nodes = sorted_nodes.astype("int32")
    sorted_positions = {i:idx for idx, i in enumerate(sorted_nodes)}
    _, min_y = sink_pos
    _, max_y = source_pos


    pos = {source: source_pos, sink: sink_pos}

    ## Setting Y positions
    #define levels
    tmp_graph = nx.Graph(graph) #Changing from diGraph to Graph
    reach_source = nx.shortest_path_length(tmp_graph, source=source)
    reach_sink = nx.shortest_path_length(tmp_graph, source=sink)

    #reach_node_lengths = {i: reach_source[i] - reach_sink[i] for i in sorted_nodes if i not in [sink, source]}
    node_levels = np.array([reach_source[i] - reach_sink[i] for i in sorted_nodes])
    level_names = set(node_levels)
    level_number = len(level_names)

    node_levels[sorted_positions[sink]] = max(node_levels) + 1 
    node_levels[sorted_positions[source]] =  min(node_levels) - 1
    level_step = -1*(max_y - min_y)/(level_number + 1)
    
    #Set y pos
    #tmp_pos_y = {node: level*level_step for node, level in reach_node_lengths.items()}
    y_pos = np.array([node_levels[sorted_positions[node]]*level_step for node in sorted_nodes])
    ## Setting X positions
    x_pos = np.zeros(len(sorted_nodes))
    nodes_on_levels = {i:0 for i in level_names}
    paths = [i.split("->") for i in msm_graph.flux.nodes.keys()]

    added_nodes = [source, sink]
    for path in paths:
        for node in path:
            node = int(node)
            if node not in added_nodes:
                level = node_levels[sorted_positions[node]]
                tmp_nodes_on_level = int(nodes_on_levels[level]) 
                sign = 2 * (tmp_nodes_on_level % 2) - 1 if  tmp_nodes_on_level != 0 else 0
                tmp_nodes_on_level = tmp_nodes_on_level if tmp_nodes_on_level % 2 == 0 else tmp_nodes_on_level + 1
                pad = 0.5 * x_step if int(np.sum(node_levels==level)) % 2 == 0 else 0
                #pos[node] = (sign * x_step * int(tmp_nodes_on_level/2) - pad, tmp_pos_y[node])
                x_pos[[sorted_positions[node]]] = sign * x_step * int(tmp_nodes_on_level/2) - pad 
                nodes_on_levels[level] += 1
                added_nodes.append(node)
    
    stair_factor = 0.1
    

    for i in level_names:
        tmp_nodes = sorted_nodes[node_levels == i]
        tmp_nodes_x = x_pos[node_levels == i]
        x_sort = np.argsort(tmp_nodes_x)
        tmp_nodes_idx = [sorted_positions[i] for i in tmp_nodes[x_sort]]
        stair_modifier = -np.arange(0, len(tmp_nodes_x)*stair_factor, stair_factor) + stair_factor * int(len(tmp_nodes_x)/2)
        y_pos[tmp_nodes_idx] += stair_modifier

    pos = {i:(x_pos[sorted_positions[i]], y_pos[sorted_positions[i]]) for i in sorted_nodes}
    return pos

    # def findPath(self, source=[], target=[]):
    #     # all_paths = nx.shortest_path(self.Gc)
    #     # all_paths = nx.dijkstra_path(self.Gc)
    #     out = {}
    #     for s in source:
    #         tmp_path = {}
    #         for t in target:
    #             try:
    #                 tmp_path[t] = nx.dijkstra_path(self.Gc, s, t)
    #             except:
    #                 pass
    #         if tmp_path != {}:
    #             out[s] = tmp_path
    #     return out

    # def plotPathwayGraph(self, paths, pos=None, transition=None, weigthed=False, **kwds):
    #     """ Creates a network representation the MSM.

    #     Draw the graphs highlighting the selected paths.

    #     Parameters
    #     ----------
    #     paths : array
    #             Array of paths leading from sources to target nodes to be plotted
    #     pos : [float]
    #             Array with the position of the nodes
    #     transition: boolean
    #             Array with the data to draw the transition 
    #     weigthed: boolean
    #             Whether to draw taking into account the eq. population of the macrostate
    #     """

    #     try:
    #         ax
    #     except:
    #         f, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))

    #     if not pos:
    #         pos = self._set_node_position(self.Gc, self.model, weighted=weigthed)
        
    #     subgraphs = [ self.Gc.subgraph(path) for path in paths ]

    #     temp_min_rate = self.min_rate
    #     self.modelToNetwork(0.1) ##Calculating graph with less nodes

    #     for subgraph in subgraphs:
    #         self.Gc = nx.compose(self.Gc, subgraph)

    #     small_pos = self._set_node_position(self.Gc, self.model, weighted=weigthed)

    #     sub_pos = { node:pos[node] for node in self.Gc.nodes }

    #     self.plotGraph(ax=ax,
    #         pos=small_pos,
    #         node_color="red",
    #         node_size= 0.5, 
    #         returnValues=True,
    #         edge_color='gray', 
    #         alpha=0.9,
    #         legend=False,
    #         arrows=False)

    #     for path in paths:
    #         subgraph = self.Gc.subgraph(path)
    #         full_path = [(path[i], path[i+1]) for i, z in enumerate(path[0:-1])]

    #         sub_position = { node:pos[node] for node in path }

    #         node_color = "blue"
    #         if transition:
    #             node_color = []
    #             for i in subgraph.nodes:
    #                 if(i == transition['bulk']):
    #                     node_color.append(transition['bulk_color'])
    #                 else:
    #                     node_color.append(transition['color_by_macro'][self.model.macro_ofmicro[i]])
            
    #         nx.draw_networkx_nodes(subgraph, 
    #             pos=sub_position, 
    #             node_size=300, 
    #             node_color=node_color, 
    #             width=3, 
    #             ax=ax)
    #         nx.draw_networkx_labels(subgraph,           
    #             pos=sub_position,
    #             font_color="white",
    #             ax=ax) 
    #         nx.draw_networkx_edges(subgraph,
    #             edgelist=full_path, 
    #             pos=sub_position, 
    #             width=3, 
    #             ax=ax)

    #     self.modelToNetwork(temp_min_rate)

    # def plotGraph(self, rate=0.1, pos=None, node_color=None, node_size=None, weigthed=False,
    #     legend=True, legend_label=None, legend_color=None, label=False,
    #     top_transition=11, returnValues=False, plot=True, **kwds):
    #     """Plots the microstate network of the model

    #     Creates a graph representation of the markov state model using as nodes 
    #     the microstates and the transition between them as the edges
    #     Executed at initializing the IDPModel

    #     Parameters
    #     ----------
    #     color : []
    #         Array of node colors
    #     position : [] 
    #         Array with the positions of each node. Calculated by default setting the most populated
    #         node in the center and the rest with a distance to the center proporcional to their connectivity degree.
    #     node_color: 
    #     weigthed : bool
    #         Whether to divide the graph into section represenative of macro population or not.
    #     top_transition : int
    #         A number bigger than one indicating the number of edges accounting with more transition
    #         in the network to higlight
    #     """

    #     # cc = nx.closeness_centrality(self.Gc, distance='weight')

    #     if not pos:
    #         all_pos = self._set_node_position(self.Gc, self.model, weighted=weigthed)
    #     else:
    #         all_pos = pos

    #     # width, edge_color = self._getPlotWidthEdges(self.Gc, self.model, top_transition=top_transition)


    #     tmp_min_rate = self.min_rate

    #     self.modelToNetwork(rate)

    #     if not node_color:
    #         node_color = self._getPlotColor(self.model, self.Gc)
        
    #     if not node_size:
    #         # node_size =  self._getMicroPop(self.model, self.Gc)
    #         node_size = self.model.msm.stationary_distribution[self.Gc.nodes]
        
    #     pos = { node: all_pos[node] for node in self.Gc.nodes }
    #     # [print(cc[i]) for i in self.Gc.nodes]
    #     if plot:
            
    #         try:
    #             ax
    #         except: 
    #             plt.figure(figsize=(8,8))

    #         nx.draw(self.Gc, node_color=node_color, 
    #             pos=pos, node_size=node_size*30000, 
    #             # edge_color=edge_color, width=width,
    #             edge_cmap=plt.cm.Blues, with_labels=label, **kwds)

    #         if legend:
    #             custom_lines = [Line2D([0], [0], color=[1, 0, 1], lw=5, ),
    #                                             Line2D([0], [0], color=[0, 0, 1], lw=5, ),
    #                                             Line2D([0], [0], color=[0, 1, 1], lw=5, )]
    #             label = ['Beta', 'Random Coil', 'Helix']
    #             if legend_color:
    #                 custom_lines = [Line2D([0], [0], color=i, lw=5) for i in legend_color]

    #             if legend_label:
    #                 label = legend_label

    #             try:
    #                 ax.legend(custom_lines, label, prop={'size': 10}, loc="upper right")
    #             except: 
    #                 plt.legend(custom_lines, label, prop={'size': 10})

    #     self.modelToNetwork(tmp_min_rate)
    #     if returnValues:
    #         return pos, node_size, node_color

    # def _angles_range(self, model):
    #     angle_value = [pop*360 for macronum, pop in zip(range(model.macronum), model.eqDistribution(plot=False)) ]
    #     angle_range = []
    #     for idx, i in enumerate(angle_value):
    #         if idx == 0:
    #             angle_range.append([0, i])
    #         else:
    #             last_value = angle_range[-1][-1]
    #             angle_range.append([ last_value, last_value + i ])
    #     return np.array(angle_range)

    # def _set_node_position(self, graph, model, max_radius=5, weighted=False):
    #     degrees = self.Gc.degree()
    #     out = np.array([np.array([i[0] , i[1]]) for i in degrees])
    #     cc = nx.closeness_centrality(self.Gc, distance='weight')

    #     try:
    #         max_conection = np.max(out[:,1])
    #     except:
    #         print("Empty graph")
    #         return
            
    #     position = {}
    #     for i in graph.nodes:
    #         #Calculate distance from center 
    #         if degrees[i] == max_conection:
    #             position[i] = np.array([0, 0])
    #             continue
    #         else:
    #             # distance_from_center = max_radius - max_radius * degrees[i]/max_conection
    #             distance_from_center = (1 - cc[i]) * max_radius
    #         #Select angle 
    #         macro = model.macro_ofmicro[i]
    #         if weighted:
    #             angle_range = self._angles_range(model)
    #             angle = np.random.randint(angle_range[macro][0] + 2, angle_range[macro][1] - 2)
    #         else:
    #             angle = np.random.randint(macro*360/model.macronum + 2, (macro + 1)*360/model.macronum - 2)                
    #         #Calculate coordinates
    #         x = distance_from_center * np.sin(np.radians(90 - angle))
    #         y = distance_from_center * np.sin(np.radians(angle))
    #         #Set position
    #         position[i] = np.array([x, y])

    #     return position

    # def _getPlotWidthEdges(self, graph, model, type="both", top_transition=10):
    #     total_w = []
    #     for node in graph.nodes:
    #         for reach_macro in graph[node]:
    #             if (node != reach_macro):
    #                 weigth = model.msm.count_matrix_active[node][reach_macro] + model.msm.count_matrix_active[reach_macro][node]
    #                 total_w.append(weigth)

    #     width = []
    #     edge_color = []
    #     max_change = max(total_w)
    #     total_w.sort()
    #     for edge in graph.edges:
    #         weigth = model.msm.count_matrix_active[edge[0]][edge[1]] + model.msm.count_matrix_active[edge[1]][edge[0]]
    #         color = weigth/max_change
    #         if (np.any(np.array(total_w[0:top_transition*2]) == weigth)):
    #             #         l_edge_color.append([(1 - color)%1, (1 - color)%1, (1 - color)%1])
    #             edge_color.append([(1 - color)%1, 0, 0])
    #             width.append(4)
    #         else:
    #             edge_color.append([(1 - color)%1, (1 - color)%1, (1 - color)%1])
    #             width.append(2)
    #     return width, edge_color

    # def _getPlotColor(self, model, graph):
    #     node_color = {} 
    #     for idx, micro in enumerate(self.data_by_micro['ss']):
    #         ss_type_mean = [np.mean(micro == ss_type) for ss_type in [1, 2]]
    #         node_color[idx] = ss_type_mean 
    #     node_color_final = []
    #     for i in graph.node:
    #         data = node_color[i]
    #         tmp_color = [1, 1, 0]
    #         max_position = np.where(data == np.max(data))[0][0]
    #         tmp_color[max_position] = 1 - data[max_position]*2
    #         node_color_final.append(tmp_color)
    #     return node_color_final