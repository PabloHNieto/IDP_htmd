"""Summary
"""
import numpy as np


class FluxController():
    """Class for controlling the flux between macrostates
    Makes use of pyemma.tpt module for the calculations of fluxes
    
    Attributes
    ----------
    model : <htmd.model.Model>
        Model to be used in the calculation of kinetics
    nodes : dict
        Dictionary containing the paths leading from the source to the sink state
    pathfluxes : np.ndarray
        Fluxes associated with each one of the paths
    paths : np.ndarray
        All paths leading from source to sink, in microstates
    save_dir : string
        Path where to store derived data
    sink : int
        Macrostate used as sink
    source : int
        Macrostates used as source
    """


    def __init__(self, paths=None, pathfluxes=None, model=None, fraction=0.9,
                 sink=None, source=None, statetype="macro", nodes=None):
        self.model = model
        self.newsets = None
        self.nodes = nodes
        self.nodes_fluxes = None
        self.net_flux = None
        self.paths = paths
        self.pathfluxes = pathfluxes
        self.sink = sink
        self.source = source
        self.statetype = statetype
        self.tpt = None
        self.fraction = fraction
        
        if (model and (isinstance(sink, int) or isinstance(sink, list)) and 
            (isinstance(source, int) or isinstance(sink, list)) and
            paths is None and pathfluxes is None):
            print("Calculating rates")
            self.paths, self.pathfluxes, self.newsets = self.calculate_in_out_rates(True)
        
        if (not self.nodes and self.model and isinstance(self.paths, list) 
                and isinstance(self.pathfluxes, list)):
            print("Translating to rates")
            self.nodes = self.translate_paths()

    def get_flux_from_path(self, paths, print_paths=False):
        """Calculates the accumulated flux for a set of paths
        
        Parameters
        ----------
        paths : [string]
            Array of paths to be matche
        print_paths : bool, optional
            If true, it will print the paths found
        
        Returns
        -------
        np.ndarray
            Array containig the percentage of flux involved in a given path
        """
        import re 
        if isinstance(paths, str):
            paths = [paths]

        total_flux = np.sum(self.pathfluxes)
        all_fluxes = []

        for path in paths:
            all_sum = 0
            for i in self.nodes:
                if (self.nodes[i] > 0 and (re.findall(path, i))):
                    if print_paths:
                        print("{0:<40s} {1:6.2f}".format(i, self.nodes[i]))
                    all_sum += self.nodes[i]
            
            all_fluxes.append(all_sum)
        return np.array(all_fluxes / total_flux, dtype="float64")


    def calculate_in_out_rates(self, coarse=False):
        from pyemma import msm
        from IDP_htmd.model_utils import metastable_states

        try:
            self.model.metastable_sets
            lookup = self.model.set_ofmicros
        except:
            print("Recalculating metastable sets")
            metastable_states(self.model)
            lookup = self.model.macro_ofmicro
        

        if self.statetype == "macro":
            if (self.sink < 0  or self.sink > len(self.model.metastable_sets) 
                or self.source < 0 or self.source > len(self.model.metastable_sets)):
                raise Exception("Sink or source out of bounds")
            tpt = msm.tpt(self.model.msm,
                        self.model.metastable_sets[self.source],
                        self.model.metastable_sets[self.sink])
        elif self.statetype == "micro":
            tpt = msm.tpt(self.model.msm,
                        self.source,
                        self.sink)

        if coarse:
            newsets, tpt = tpt.coarse_grain(self.model.metastable_sets)
            paths, pathfluxes = tpt.pathways(fraction=self.fraction)

            # Create a lookup table for new datasets
            if self.statetype == "macro":
                macro2macro = np.zeros(len(self.model.metastable_sets), dtype=int)
                for idx, micro in enumerate(newsets):
                    macro2macro[idx] = lookup[micro[0]]
            elif self.statetype == "micro":
                macro2macro = np.zeros(len(newsets), dtype=int)
                for idx, micro in enumerate(newsets):
                    macro2macro[idx] = lookup[micro[0]]
        else:
            paths, pathfluxes = tpt.pathways(fraction=self.fraction)
            macro2macro = lookup
        
        self.tpt = tpt
        return paths, pathfluxes, macro2macro

    def translate_paths(self):
        """Translate paths from micro to macros jumps
        
        Returns
        -------
        dict
            Dictionary containing each path leading from source to sink and its accumulated flux
        
        Deleted Parameters
        ------------------
        paths : np.ndarray
            Paths leading from source to sink
        pathfluxes : np.ndarray
            Fluxes associated with each path
        model : <htmd.model.Model>
            Model used to calculte the fluxes
        """

        nodes = {}

        for path, flux in zip(self.paths, self.pathfluxes):
            if self.newsets is None:
                tmp = [str(self.model.macro_ofmicro[micro]) for micro in path]
                tmp_2 = []
                #Removing consecutive identical values
                for idx2, _ in enumerate(tmp[0:-1]):
                    if idx2 == 0:
                        tmp_2.append(tmp[idx2])
                    if idx2 > 0 and tmp[idx2] != tmp[idx2 + 1]:
                        tmp_2.append(tmp[idx2 + 1])
                tmp = "->".join(tmp_2)
            else:
                tmp = "->".join([str(self.newsets[i]) for i in path])

            if tmp in nodes.keys():
                nodes[tmp] += flux
            else:
                nodes[tmp] = flux
        return nodes

    def calculate_transition_rates(self):
        source, sink = self.source, self.sink
        if self.statetype == "macro":
            start_paths = [f'^{source}->{i}->' for i in range(len(self.model.metastable_sets))]
            end_paths = [f'->{i}->{sink}$' for i in range(len(self.model.metastable_sets))]
            all_paths = [f'->{i}->' for i in range(len(self.model.metastable_sets))]
            
        elif self.statetype == "micro":
            start_paths = [f'^{so}->{i}->' for i in range(self.model.micronum) for so in source]
            end_paths = [f'->{i}->{si}$' for i in range(self.model.micronum) for si in sink]
            all_paths = [f'->{i}->' for i in range(self.model.micronum)]
        
        start_fluxes = self.get_flux_from_path(start_paths)
        end_fluxes = self.get_flux_from_path(end_paths)
        all_fluxes = self.get_flux_from_path(all_paths)

        return start_fluxes, end_fluxes, all_fluxes
  
    def plot_in_out_rates(self, labels=None, save=None, strict=True, thr=0):
        """Plot in and out rates.
        In rate: % of the flux that reaches each macrosates after jumping from the source
        Out rate: % of the flux that directly reaches the sink from each other macrostate

        Parameters
        ----------
        labels : [type], optional
            [description] (the default is None, which [default_description])
        save : [type], optional
            [description] (the default is None, which [default_description])
        """

      
        source, sink = self.source, self.sink

        start_paths = [f'^{source}->{i}->' for i in range(len(self.model.metastable_sets))]
        end_paths = [f'->{i}->{sink}$' for i in range(len(self.model.metastable_sets))]
        all_paths = [f'->{i}->' for i in range(len(self.model.metastable_sets))]
        
        start_fluxes = self.get_flux_from_path(start_paths)
        end_fluxes = self.get_flux_from_path(end_paths)
        all_fluxes = self.get_flux_from_path(all_paths) 

        #start_fluxes, end_fluxes, all_fluxes = self.calculate_transition_rates()

        if labels:
            end_fluxes = np.array([x for y, x in sorted(zip(end_paths, end_fluxes), 
                                                key=lambda pair: len(pair[0]))])
            start_fluxes = np.array([x for y, x in sorted(zip(start_paths, start_fluxes), 
                                                key=lambda pair: len(pair[0]))])
        else:
            labels = range(len(self.model.metastable_sets))

        #When calculating intermediate fluxes, direct fluxes will be rested twice. Avoid this.
        direct_macros_fluxes = [self.get_flux_from_path([f"{source}->{i}->{sink}"])[0] for i in range(len(self.model.metastable_sets))]
        intermediate_fluxes = all_fluxes - start_fluxes - end_fluxes + direct_macros_fluxes

        selected_macros = np.where((end_fluxes > thr) | (start_fluxes > thr) | (intermediate_fluxes > thr))[0]
        
        import matplotlib.pyplot as plt

        tmp_end_fluxes = end_fluxes[selected_macros]
        tmp_start_fluxes = start_fluxes[selected_macros]
        tmp_intermediate_fluxes = intermediate_fluxes[selected_macros]
        labels = np.arange(len(self.model.metastable_sets))[selected_macros]

        c1 = (0, 0, 1)
        c2 = "C1"
        c3 = (0.8, 0, 0)
        from matplotlib.cm import get_cmap
        cmap = get_cmap("Set1", 9)
        c1 = cmap(0)
        c2 = cmap(1)
        c3 = cmap(2)
        
        plt.bar(list(range(len(tmp_start_fluxes))), tmp_start_fluxes * 100, color=c1, 
                edgecolor=c1, alpha=0.8, label="In")
        plt.bar(list(range(len(tmp_start_fluxes))), tmp_intermediate_fluxes * 100, color=c2, 
                edgecolor=c2, alpha=0.8, label="Between")
        plt.bar(list(range(len(tmp_start_fluxes))), tmp_end_fluxes * 100, color=c3, 
                edgecolor=c3, alpha=0.8, label="Out")

        _, _ = plt.xticks(list(range(len(tmp_start_fluxes))), labels, rotation='vertical', ha="center")
        _ = plt.legend(loc='upper center', shadow=True, ncol=3)

        plt.title(f"In & Out rates from {self.statetype.capitalize()}-{self.source} to {self.statetype.capitalize()}-{self.sink}")
        plt.xlabel("Macrostate")
        plt.ylabel("Flux (%)")
        plt.xticks(rotation=45)

        if save:
            plt.savefig(save, dpi=200, bbox_inches='tight', pad_inches=0.2)

        return start_fluxes, end_fluxes, all_fluxes, selected_macros

    @staticmethod
    def rates_comparison(f0, f1, plot=True, save=None):
        """Summary
        
        Parameters
        ----------
        f0 : TYPE
            Description
        f1 : TYPE
            Description
        plot : bool, optional
            Description
        save : None, optional
            Description
        """
        import matplotlib.pyplot as plt
        start_f0_paths = [f'{f0.source}->{i}->' for i in range(len(f0.model.metastable_sets))]
        start_f1_paths = [f'{f1.source}->{i}->' for i in range(len(f1.model.metastable_sets))]
        end_f0_paths = [f'->{i}->{f0.sink}'.format(i) for i in range(len(f0.model.metastable_sets))]
        end_f1_paths = [f'->{i}->{f1.sink}'.format(i) for i in range(len(f1.model.metastable_sets))]

        start_f0_flux = f0.get_flux_from_path(start_f0_paths, print_paths=False) * 100
        start_f1_flux = f1.get_flux_from_path(start_f1_paths, print_paths=False) * 100
        end_f0_flux = f0.get_flux_from_path(end_f0_paths, print_paths=False) * 100
        end_f1_flux = f1.get_flux_from_path(end_f1_paths, print_paths=False) * 100

        plt.figure(figsize=(7, 7))

        plt.barh(list(range(len(start_f0_flux))), start_f0_flux[::-1], color=(0, 0, 1, 0.4), 
                 edgecolor=(0, 0, 0.5, 0.5), label="In")
        plt.barh(list(range(len(start_f0_flux))), end_f0_flux[::-1], color=(0.8, 0, 0, 0.2), 
                 edgecolor=(0.5, 0, 0, 0.5), label="Out")
        plt.barh(list(range(len(start_f1_flux))), -1 * start_f1_flux[::-1], color=(0, 0, 1, 0.4),
                 edgecolor=(0, 0, 0.5, 0.5))
        plt.barh(list(range(len(end_f1_flux))), -1 * end_f1_flux[::-1], color=(0.8, 0, 0, 0.2), 
                 edgecolor=(0.5, 0, 0, 0.5))

        y_limit = np.max([f0.model.macronum, f1.model.macronum])
        plt.ylim((-.5, y_limit + .5))
        _, _ = plt.yticks(list(range(len(f1.model.metastable_sets))), list(range(len(f1.model.metastable_sets)))[::-1])
        upper_bound = np.max(np.array([start_f0_flux, start_f1_flux, end_f1_flux, end_f0_flux])) + 5
        plt.xlim((-1 * upper_bound, upper_bound))
        plt.vlines(0, -1, 16, color=(0, 0, 0, 0.7))
        plt.grid(axis='x')

        plt.title("In & Out intermediate Macros")
        plt.ylabel("Macrostates Label")
        plt.xlabel("Bound (%)                   Bound (%)")

        plt.legend(loc='upper right', shadow=True)

        if plot:
            plt.show()

        if save:
            plt.savefig(save, dpi=200, bbox_inches='tight', pad_inches=0.2)


if __name__ == "__main__":
    pass
