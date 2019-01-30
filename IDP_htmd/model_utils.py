import numpy as np
from sklearn.cluster import MiniBatchKMeans
from htmd.molecule.molecule import Molecule
from htmd.model import Model, getStateStatistic


def get_params_model(model):
    return {'clusters': len(model.micro_ofcluster),
            'lag': int(round(model.lag * model.data.fstep, 0)),
            'macroN': model.macronum}


def save_structures(model, outdir, states, numsamples, statetype,
                    modifications=None, **kwargs):
    import os
    from glob import glob
    if len(states) != len(numsamples) and len(numsamples) != len(statetype):
        print("Length of states, numsamples and statetype should match")
        return
    os.makedirs(outdir, exist_ok=True)
    for idx, i in enumerate(states):
        m = model.getStates(statetype=statetype[idx], states=[i],
                            numsamples=int(numsamples[idx]), **kwargs)
        for struct in m:
            if modifications:
                for prop, setting, sel in modifications:
                    struct.set(prop, setting, sel)
            for frame in range(struct.numFrames):
                out_name = f"{outdir}/{statetype[idx]}_{i}_{frame}.pdb"
                struct.frame = frame
                struct.resname[struct.resname == 'HSD'] = "HIS"
                # struct.write(out_name, sel="not name CAY CY OY NT CAT")
                struct.write(out_name, sel="protein")
    return glob(outdir+"/*pdb")


def get_weighted(model, total_struct):
    population = model.eqDistribution(plot=False)
    out_structs = np.array([int(total_struct * pop) for pop in population])

    idx_max = np.where(population == np.max(population))
    if np.sum(out_structs) < total_struct:
        out_structs[idx_max] += total_struct - np.sum(out_structs)
    elif np.sum(out_structs) > total_struct:
        out_structs[idx_max] -= total_struct - np.sum(out_structs)
    return out_structs


def metastable_states(model):
    metastable_sets = []
    for i in range(model.macronum):
        metastable_sets.append(np.where(model.macro_ofmicro == i)[0])
    model.metastable_sets = np.array(metastable_sets)


def get_data(model, metr, skip=1):
    """ Returns the projected data of metric applied to a model

        Parameters
        ----------
        mod : htmd.model.Model
            Model to get the simlist
        metric : htmd.projections.MetricData
            MetricData with the metric we want to project
        skip : int
            Frames to skip while projecting the data. Default = 1
        """
    from htmd.projections.metric import Metric
    metric = Metric(model.data.simlist, skip=skip)
    metric.set(metr)
    data = metric.project()
    return data


def create_bulk(model, metric=None, data=None):
    """Creates a bulk macrosates
    Modifies passed model
    It is intended to be used in ligand binding escenarios.
    
    Parameters
    ----------
    model : TYPE
        Model to extract a bulk
    metric : TYPE
        Metric to describe a bulk vs not-bulk situation. In general is the contacts 
        between protein and ligand selection with groupsels set to 'all'
    data : None, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    
    Raises
    ------
    Exception
        Description
    """
    if not metric and not data:
        raise Exception("Either a metric or a data object must be provided")
    
    if metric and not data:
        data = get_data(model, metric)

    data_by_micro = np.array(getStateStatistic(model, data, 
                                               states=range(model.micronum), statetype="micro"))
    min_contacts = np.where(data_by_micro == np.min(data_by_micro))[0]
    model.createState(min_contacts)
    print(f"Macrostate created with micros: {min_contacts}")
    return min_contacts


def cluster_macro(model, data, macro, method=np.mean, cluster_method=MiniBatchKMeans):
    """Modifies the model by splitting a macrostate.
    In first place, the mean for the given data is calculated for each micro
    of the model. This data is then clustered using the MiniBatchKMeans algorithm
        
    Parameters
    ----------
    model : <htmd.model.Model>
        Model to be modified
    data : TYPE
        Description
    macro : int
        Macrostate to be splitted
    method : TYPE, optional
        Description
    """
    # from sklearn.cluster import MiniBatchKMeans, AffinityPropagation
    from IDP_htmd.IDP_model import plot_contacts

    if macro < 0 or macro > model.macronum:
        raise Exception("Macro out of bounds")
    mol = Molecule(model.data.simlist[0].molfile)
    data_by_micro = getStateStatistic(model, data, states=model.metastable_sets[macro], 
                                      statetype="micro", method=method)
    clusters = cluster_method().fit(data_by_micro)

    for i in range(len(clusters.cluster_centers_) - 1):
        label_micro = model.metastable_sets[macro][np.where(clusters.labels_ == i)[0]]
        # cluster = getStateStatistic(model, data,
        #                             states=label_micro, statetype="micro", method=np.mean)
        # labels = ['Micro {}'.format(i) for i in label_micro]
        # plot_contacts(cluster, mol, labels=labels, title=f"Cluster {i}", 
        #               plot=True, 
        #               # save=f"/home/pablo/test_info/{i}_plt_contacts.png"
        #               )
        model.createState(label_micro)

def compute_all_mfpt(model):
    """Calculates the mean first passages time in (ns) between
    every macrostates within an MSM.
    
    Parameters
    ----------
    model : <htmd.model.Model>
        Model which mfpt will be computed
    
    Returns
    -------
    mfpt: np.ndarray
        Matrix with the mfpt between states: 
        "from... Macro of row index to... Macro of column index"
    """
    all_mfpt = []
    for source in range(model.macronum):
        all_mfpt.append([model.msm.mfpt(source, sink) for sink in range(model.macronum)])
    return np.array(all_mfpt)

def scan_clusters(model, nclusters, out_dir):
    """Create models 
    
    In order to assess the effect on timescales using different clusters in a model.
    Parameters
    ----------
    model : htmd.model.Model
        Model class we want to perfom the analysis
    nclusters : int[]
        Array of clusters to be tested
    out_dir : str
        Directory to save the generated plots
    """
    from sklearn.cluster import MiniBatchKMeans
    for i in nclusters:
        model.data.cluster(MiniBatchKMeans(n_clusters=i), mergesmall=5)
        new_mod = Model(model.data)
        new_mod.plotTimescales(plot=False, save=out_dir+"1_its-{}_clu".format(i))


def aux_plot(model, mol, plot_func, metric=None, skip=1, normalize=False, method=np.mean, data=None, **kwargs):
    """Summary

    Parameters
    ----------
    model : TYPE
        Model to extract the data
    metric : TYPE
        Metric object to project the simlist of the model
    mol : TYPE
        Description
    plot_func : TYPE
        Plotting function to plot the projected data
    skip : int, optional
        Skip frames from the simlist
    normalize : bool, optional
        Whether to normalize by the number of atoms
    method : TYPE, optional
        Method to perform the aggregation of the data by macrostate
    **kwargs
        Additional arguments for the plotting function
    """
    if not metric and not data:
        raise Exception("Either a metric or a data object must be provided")
        
    if not data:
        data = get_data(model, metric, skip=skip)

    data_summary = getStateStatistic(model, data, method=method, states=range(model.macronum),
                                     statetype="macro")

    if normalize:
        _, counts = np.unique(mol.resid, return_counts=True)
        data_summary = np.array(data_summary) / counts

    try:
        plot_func(data_summary, mol, **kwargs)
    except Exception as e:
        print("Plotting error: ", e)


def bootstrap(model, rounds, fraction=0.8, clusters=500):
    import os
    from htmd.model import Model
    from sklearn.cluster import MiniBatchKMeans

    for boot_round in range(rounds):
        dataBoot = model.data.bootstrap(fraction)
        print(f"Starting a new round of bootstrap - {boot_round}")
        dataBoot.cluster(MiniBatchKMeans(n_clusters=clusters), mergesmall=5)
        model = Model(dataBoot)
        yield(model)


if __name__ == "__main__":
    model = Model()
    base_dir = "/workspace8/p27_sj403/27-10-2018_p27_sj403/"
    model.load(f"{base_dir}final_model_split.dat")
    p, pf = calculate_in_out_rates(model, [1,2], 3)