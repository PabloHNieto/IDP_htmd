import numpy as np


class FluxController():
    def __init__(self, paths=None, pathfluxes=None, model=None, 
                out_macro=None, bulk=None, nodes=None, save_dir=None):
        self.paths = paths
        self.pathfluxes = pathfluxes
        self.model = model
        self.sink = out_macro
        self.source = bulk
        self.nodes = nodes
        self.save_dir = save_dir
        
        if self.model and self.source and self.sink and not self.paths and not self.pathfluxes:
            print("Calculating rates")
            self.paths, self.pathfluxes = self.calculate_in_out_rates(self.model, self.source, self.save_dir)
        
        if not self.nodes and self.model and self.paths and self.pathfluxes:
            self.nodes = self.translate_paths(self.paths, self.pathfluxes, self.model)

    def get_flux_from_path(self, paths, fluxes, print_paths=True):
        total_sum = 0
        all_fluxes = []
        for path in paths:
            all_sum = 0
            for i in fluxes:
                if (fluxes[i]>0
                    and (path in i)):
                    if print_paths:
                        print("{0:<40s} {1:6.2f}".format(i, fluxes[i]))
                    all_sum += fluxes[i]
            all_fluxes.append(all_sum)
            total_sum += all_sum
        return np.array(all_fluxes)
    

    def calculate_in_out_rates(self, model, sink, source, save_dir=None):
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
            Path where to save the data (the default is None, which does not save the path and pathfluxes)
        
        Returns
        -------
        [np.ndarray]
            Array of same length as sinks argument. For each sink it return the path and their fluxes
        """
        from pyemma import msm
        from IDP_htmd.model_utils import metastable_states

        metastable_states(model)
        
        # out = []
        # for i in out_macros:

        if sink < 0  or sink > model.macronum or source < 0 or source > model.macronum:
            raise Exception("Sink or source out of bounds")

        tpt = msm.tpt(model.msm, model.metastable_sets[source], model.metastable_sets[sink])
        paths, pathfluxes = tpt.pathways(fraction=0.9)

        if save_dir:
            np.save(f"{save_dir}/path_{sink}_fluxes.npy", np.array([paths, pathfluxes]))
        return paths, pathfluxes


    def load_paths(self, model, out_macro, bulk_macro, filename):
        if filename:
            path, pathfluxes = np.load(filename)
        else: 
            print("Calculating pathways")
            self.source = out_macro
            self.sink = bulk_macro
            self.model = model
            self.path, self.pathfluxes = self.calculate_in_out_rates(model, out_macro, bulk_macro)
        
        return path, pathfluxes
        

    def translate_paths(self, paths, pathfluxes, model):
        # Translate paths from micro to macros jumps
        nodes = {}
        for idx, (path, flux) in enumerate(zip(paths, pathfluxes)):
            tmp = [str(model.macro_ofmicro[micro]) for micro in path ]
            tmp_2 = []
            #Removing consecutive identical values    
            for idx, _ in enumerate(tmp[0:-1]):
                if idx == 0:
                    tmp_2.append(tmp[idx])
                if idx > 0 and tmp[idx] != tmp[idx + 1]:
                    tmp_2.append(tmp[idx + 1])

            tmp = "->".join(tmp_2)
            if tmp in nodes.keys():
                nodes[tmp] += flux
            else:
                nodes[tmp] = flux
        return nodes

    def plot_in_out_rates(self, model, labels=None, plot=False, save=None):
        import matplotlib.pyplot as plt

        start_paths = ['3->{}->'.format(i) for i in range(model.macronum)]
        end_paths = ['->{}->1'.format(i) for i in range(model.macronum)]

        start_fluxes = self.get_flux_from_path(start_paths, self.pathfluxes, print_paths=False)
        end_fluxes = self.get_flux_from_path(end_paths, self.pathfluxes, print_paths=False)
        
        if labels:
            end_fluxes = [x for y, x in sorted(zip(start_paths,end_fluxes), key=lambda pair: len(pair[0]))]
            start_fluxes = [x for y, x in sorted(zip(start_paths,start_fluxes), key=lambda pair: len(pair[0]))]

        plt.bar(list(range(len(start_fluxes))), start_fluxes, color=(0, 0, 1, 0.4), edgecolor=(0,0,0.5,0.5))
        plt.bar(list(range(len(start_fluxes))), end_fluxes, color=(0.8, 0, 0, 0.2), edgecolor=(0.5,0,0,0.5))

        sorted_labels = range(model.macronum)
        _,_ = plt.xticks(list(range(len(start_fluxes))), sorted_labels, rotation='vertical', ha="center")

        if plot:
            plt.show()

        if save:
            plt.savefig(save, dpi=200, bbox_inches='tight', pad_inches=0.2)