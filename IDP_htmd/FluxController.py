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


    def __init__(self, paths=None, pathfluxes=None, model=None,
                 sink=None, bulk=None, nodes=None, save_dir=None):
        self.paths = paths
        self.pathfluxes = pathfluxes
        self.model = model
        self.sink = sink
        self.source = bulk
        self.nodes = nodes
        self.save_dir = save_dir
        
        if (self.model and self.source and self.sink and
                self.paths is None and self.pathfluxes is None):
            print("Calculating rates")
            self.paths, self.pathfluxes = self.calculate_in_out_rates()
        
        if (not self.nodes and self.model and isinstance(self.paths, np.ndarray) 
                and isinstance(self.pathfluxes, np.ndarray)):
            print("Translating to rates")
            self.nodes = self.translate_paths()


    def get_flux_from_path(self, paths, print_paths=True):
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
        total_flux = np.sum(self.pathfluxes)
        all_fluxes = []
        for path in paths:
            all_sum = 0
            for i in self.nodes:
                if (self.nodes[i] > 0 and (path in i)):
                    if print_paths:
                        print("{0:<40s} {1:6.2f}".format(i, self.nodes[i]))
                    all_sum += self.nodes[i]
            all_fluxes.append(all_sum)
        return np.array(all_fluxes / total_flux)

    def save_data(self):
        """Saved calculated fluxes
        """
        if self.save_dir:
            np.save(f"{self.save_dir}p_{self.sink}_path_fluxes.npy", 
                    np.array([self.paths, self.pathfluxes]))


    def calculate_in_out_rates(self):
        """[summary]
        
        Parameters
        ----------
        model : <htmd.model.Model>
            Model to be used to calculate the fluxes and the paths
        sink : int
            Integer of the sink state
        source : int
            Macrostates used as source
        save_dir : string, optional
            Path where to save the data (the default is None, 
            which does not save the path and pathfluxes)
        
        Returns
        -------
        [np.ndarray]
            Array of same length as sinks argument. 
            For each sink it return the path and their fluxes
        """
        from pyemma import msm
        from IDP_htmd.model_utils import metastable_states

        metastable_states(self.model)
        
        if (self.sink < 0  or self.sink > self.model.macronum 
                or self.source < 0 or self.source > self.model.macronum):
            raise Exception("Sink or source out of bounds")

        tpt = msm.tpt(self.model.msm,
                      self.model.metastable_sets[self.source],
                      self.model.metastable_sets[self.sink])
        paths, pathfluxes = tpt.pathways(fraction=0.9)

        if self.save_dir:
            np.save(f"{self.save_dir}/path_{self.sink}_fluxes.npy", np.array([paths, pathfluxes]))
        return paths, pathfluxes


    def load_paths(self, model, out_macro, bulk_macro, filename):
        """Summary
        
        Parameters
        ----------
        model : TYPE
            Description
        out_macro : TYPE
            Description
        bulk_macro : TYPE
            Description
        filename : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        if filename:
            path, pathfluxes = np.load(filename)
        else: 
            print("Calculating pathways")
            self.source = out_macro
            self.sink = bulk_macro
            self.model = model
            path, pathfluxes = self.calculate_in_out_rates()
        
        return path, pathfluxes


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
        for idx, (path, flux) in enumerate(zip(self.paths, self.pathfluxes)):
            tmp = [str(self.model.macro_ofmicro[micro]) for micro in path]
            tmp_2 = []
            #Removing consecutive identical values
            for idx2, _ in enumerate(tmp[0:-1]):
                if idx2 == 0:
                    tmp_2.append(tmp[idx])
                if idx2 > 0 and tmp[idx] != tmp[idx2 + 1]:
                    tmp_2.append(tmp[idx + 1])

            tmp = "->".join(tmp_2)
            if tmp in nodes.keys():
                nodes[tmp] += flux
            else:
                nodes[tmp] = flux
        return nodes


    def plot_in_out_rates(self, labels=None, plot=True, save=None):
        """Plot in and out rates.
        In rate: % of the flux that reaches each macrosates after jumping from the source
        Out rate: % of the flux that directly reaches the sink from each other macrostate

        Parameters
        ----------
        labels : [type], optional
            [description] (the default is None, which [default_description])
        plot : bool, optional
            [description] (the default is False, which [default_description])
        save : [type], optional
            [description] (the default is None, which [default_description])
        """
        import matplotlib.pyplot as plt

        start_paths = [f'{self.source}->{i}->' for i in range(self.model.macronum)]
        end_paths = [f'->{i}->{self.sink}'.format(i) for i in range(self.model.macronum)]

        start_fluxes = self.get_flux_from_path(start_paths, print_paths=False)
        end_fluxes = self.get_flux_from_path(end_paths, print_paths=False)
        
        if labels:
            end_fluxes = [x for y, x in sorted(zip(start_paths, end_fluxes), 
                                               key=lambda pair: len(pair[0]))]
            start_fluxes = [x for y, x in sorted(zip(start_paths, start_fluxes), 
                                                 key=lambda pair: len(pair[0]))]
        else:
            labels = range(self.model.macronum)

        plt.bar(list(range(len(start_fluxes))), start_fluxes * 100, color=(0, 0, 1, 0.4), 
                edgecolor=(0, 0, 0.5, 0.5), label="In")
        plt.bar(list(range(len(start_fluxes))), end_fluxes * 100, color=(0.8, 0, 0, 0.2), 
                edgecolor=(0.5, 0, 0, 0.5), label="Out")

        _, _ = plt.xticks(list(range(len(start_fluxes))), labels, rotation='vertical', ha="center")
        _ = plt.legend(loc='upper right', shadow=True)

        plt.title(f"In & Out rates from Macro-{self.source} to Macro-{self.sink}")
        plt.xlabel("Macrostate")
        plt.ylabel("% Flux")
        plt.xticks(rotation=45)

        if plot:
            plt.show()

        if save:
            plt.savefig(save, dpi=200, bbox_inches='tight', pad_inches=0.2)


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
        start_f0_paths = [f'{f0.source}->{i}->' for i in range(f0.model.macronum)]
        start_f1_paths = [f'{f1.source}->{i}->' for i in range(f1.model.macronum)]
        end_f0_paths = [f'->{i}->{f0.sink}'.format(i) for i in range(f0.model.macronum)]
        end_f1_paths = [f'->{i}->{f1.sink}'.format(i) for i in range(f1.model.macronum)]

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
