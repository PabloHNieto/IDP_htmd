import numpy as np
from sklearn.cluster import MiniBatchKMeans
from moleculekit.molecule import Molecule
from htmd.model import Model, getStateStatistic
import matplotlib.pyplot as plt


def get_params_model(model):
    return {'clusters': len(model.micro_ofcluster),
            'lag': int(round(model.lag * model.data.fstep, 0)),
            'macroN': model.macronum}

def viewModel(model_name):
    model = Model(file=model_name)
    try:
        model.macronum
    except:
        model.markovModel(20, 5, units="ns")
  
    model.viewStates(alignsel="noh and resname MOL", protein=True, ligand="protein and backbone")

def save_structures(model, outdir, states, numsamples, statetype="macro",
                    modifications=None, **kwargs):
    """[summary]
        
        Parameters
        ----------
        model : htmd.model.Model
            Model to get the structuress
        outdir : str
            Folder where to save the structures
        states : []
            [description]
        numsamples : [type]
            [description]
        statetype : str, optional
            [description], by default "macro"
        modifications : [type], optional
            [description], by default None
        """
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
                try:
                    struct.write(out_name, sel="all")
                except:
                    continue
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


def metastable_states(model, modify=True):
    metastable_sets = []
    for i in range(model.macronum):
        metastable_sets.append(np.where(model.macro_ofmicro == i)[0])
    if modify: model.metastable_sets = np.array(metastable_sets)
    return metastable_sets

def set_ofmicros(model):
    try:
        model.metastable_sets
    except:
        metastable_states(model)

    model.set_ofmicros = np.zeros(model.micronum)
    for idx, i in enumerate(model.metastable_sets):
        for micro in i:
            model.set_ofmicros[micro] = idx


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
    from htmd.model import Model
    from htmd.projections.metric import Metric
    if isinstance(model, Model):
        simlist = model.data.simlist
    elif isinstance(model, np.ndarray):
        simlist = model
    else:
        raise TypeError("Model should be either an htmd.model.Model or a simlist")

    metric = Metric(simlist, skip=skip)
    metric.set(metr)
    data = metric.project()
    return data

def create_bulk(model, metric=None, data=None, threshold=0.2, skip=1):
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
        data = get_data(model, metric, skip=skip)

    data_by_micro = np.array(getStateStatistic(model, data, 
                                               states=range(model.micronum), statetype="micro"))
    min_contacts = np.where(data_by_micro<threshold)[0]

    if len(min_contacts) == 0:
        min_contacts = [np.argmin(data_by_micro<threshold)]

    model.createState(min_contacts)
    print(f"Macrostate created with micros: {min_contacts}")
    return min_contacts

def create_thresholdbulk(model, threshold=0.1, metric=None, data=None, skip=1):
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
        data = get_data(model, metric, skip=skip)

    data_by_micro = np.array(getStateStatistic(model, data, 
                                               states=range(model.micronum), statetype="micro"))
    min_contacts = np.argmin(data_by_micro)
    model.createState([min_contacts])
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
    #from IDP_htmd.IDP_model import plot_contacts

    metastable_states(model)
    if isinstance(macro, int):
        macro = [macro]

    all_micros = np.array([], dtype=int)
    for i in macro: 
        if i < 0 or i > model.macronum:
            raise Exception("Macro out of bounds")
        all_micros = np.concatenate([all_micros, model.metastable_sets[i]])
    data_by_micro = getStateStatistic(model, data, states=all_micros, 
                                      statetype="micro", method=method)
    clusters = cluster_method().fit(data_by_micro)

    new_macro_assignment = []
    for i in range(len(clusters.cluster_centers_)):
        new_macro_assignment.append(all_micros[np.where(clusters.labels_ == i)[0]])

    return np.array(new_macro_assignment)
    # for i in range(len(clusters.cluster_centers_) - 1):
    #     label_micro = model.metastable_sets[macro][np.where(clusters.labels_ == i)[0]]
    #     # cluster = getStateStatistic(model, data,
    #     #                             states=label_micro, statetype="micro", method=np.mean)
    #     # labels = ['Micro {}'.format(i) for i in label_micro]
    #     # plot_contacts(cluster, mol, labels=labels, title=f"Cluster {i}", 
    #     #               plot=True, 
    #     #               # save=f"/home/pablo/test_info/{i}_plt_contacts.png"
    #     #               )
    #     model.createState(label_micro)

def compute_all_attr(model, temperature=310, concentration=0.0026):
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
    from htmd.kinetics import Kinetics
    
    try:
        model.metastable_sets
    except:
        metastable_states(model)

    all_attr = []
    for source in model.metastable_sets:
        for sink in model.metastable_sets:
            kin = Kinetics(model, sink=model.macro_ofmicro[sink[0]], source=model.macro_ofmicro[source[0]], 
                temperature=temperature, concentration=concentration)
            all_attr.append(kin.getRates(source=source, sink=sink, states="micro"))
    return np.array(all_attr)

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
        new_mod.plotTimescales(plot=False, save=f"{out_dir}/1_its-{i}_clu")


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
    from htmd.model import Model
    from sklearn.cluster import MiniBatchKMeans

    for boot_round in range(rounds):
        dataBoot = model.data.bootstrap(fraction)
        print(f"Starting a new round of bootstrap - {boot_round}")
        dataBoot.cluster(MiniBatchKMeans(n_clusters=clusters), mergesmall=5)
        b_model = Model(dataBoot)
        yield(b_model)

def plotMSM(model, dimx=0, dimy=1, s=5, title=None, save=None, lims=[-180, 180], yellows=True, legend=True, 
        npoints=100, rewards=None, actions=None, ax=None, cmap='Set1', zorder=1, labels=None):
    import matplotlib as mpl
    
    if not ax:
        ax = plt.gca()

    if not yellows:
        def get_cmap(number): return f"C{number}"
        # def get_cmap(number):
        #     if number < model.macronum - 1 :
        #         cmap2 = mpl.cm.get_cmap(cmap, model.macronum)
        #         return cmap2(number)
        #     else:
        #         return "C1"
        new_cmap = get_cmap
    else:
        new_cmap = mpl.cm.get_cmap(cmap, model.macronum)
        
    for macro in np.argsort(model.eqDistribution(plot=False))[::-1]:
        macrocenters = model.data.Centers[np.where(model.macro_ofcluster == macro)[0], :]
        macro_pop = round(model.eqDistribution(plot=False)[macro]*100, 1)
        c = new_cmap(macro)
        if labels is not None:
            lab = labels[macro]
        else:
            lab = 'Macro {}-{}%'.format(macro, macro_pop)
        ax.scatter(macrocenters[:,dimx], macrocenters[:,dimy], alpha=0.5, edgecolor=c, s=s, color=c, label=lab, zorder=zorder)
    if legend: ax.legend()
        
    if rewards is not None:
        from sklearn.preprocessing import normalize
        norm_rewards = normalize(rewards.reshape(-1, 1), norm="max", axis=0).ravel()
        cmap = mpl.cm.get_cmap('viridis')
        colors = [cmap(i) for i in norm_rewards]
        macrocenters = model.data.Centers[model.cluster_ofmicro]
        plt.scatter(macrocenters[:,dimx], macrocenters[:,dimy], color=colors, s=s)

    if actions:    
        macrocenters = model.data.Centers[np.where(actions == 1)[0], :]
        plt.scatter(macrocenters[:,dimx], macrocenters[:,dimy], c="black", s=s)
    
    if title:
        plt.title(title, fontsize=30)

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)

def plotTica(model, dimx=0, dimy=1, fit=False, ax=None, contour=False, heatmap=True, cmap="Greys", labels=True, colorbar=False, cbar_label=None, fig=None, zorder=1):
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from htmd.model import Model
    from htmd.metricdata import MetricData
    import numpy as np

    if not fit:
        if (isinstance(model, Model)):
            tica_lin = np.concatenate(model.data.dat)
        elif (isinstance(model, MetricData)):
            tica_lin = np.concatenate(model.dat)
        elif (isinstance(model, np.ndarray)):
            tica_lin = model
    else:
        tica_lin = model
    
    # tica_lin = np.concatenate(model.data.dat)
    
    xmin, xmax = [np.min(tica_lin[:, dimx]), np.max(tica_lin[:, dimx])]
    ymin, ymax = [np.min(tica_lin[:, dimy]), np.max(tica_lin[:, dimy])]

    counts, xbins, ybins = np.histogram2d(tica_lin[:, dimx], tica_lin[:, dimy], 
                                              bins=200, range=[[xmin, xmax],[ymin, ymax]])

    xcenters = (xbins[:-1] + xbins[1:]) / 2
    ycenters = (ybins[:-1] + ybins[1:]) / 2
    counts = counts.T
    a, b = np.meshgrid(xcenters, ycenters)
    
    if ax is not None: plt.sca(ax)

    if contour:
        _ = plt.contour(a, b, counts)
    
    if heatmap:
        ims = plt.imshow(counts, interpolation='nearest', origin='lower', aspect='auto',
            extent=[xmin, xmax, ymin, ymax],
            cmap=cmap, norm=LogNorm(), zorder=zorder)

        if labels:
            plt.xlabel(f'TICA Dim. {dimx}', fontsize=12)
            plt.ylabel(f'TICA Dim. {dimy}', fontsize=12)
    
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.)
        cbar = fig.colorbar(ims, cax=cax)
        cbar.ax.set_ylabel(cbar_label)

def plotTicaSpawn(model, ax=None, last=False, cmap="viridis", s=10, alpha=1, colorbar=False, fig=None, cbar_label=None, **kwargs):
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    project_data= fitBaselineSpam(model, **kwargs)
    epochs = project_data.epoch.unique()
    
    if last:
        epochs = epochs[-last:]

    all_data_x = []
    all_data_y = []
    all_colors_idx = []
    for idx, i in enumerate(epochs):
        tmp_data_y = list(project_data.y[project_data.epoch==i])
        all_data_x += list(project_data.x[project_data.epoch==i])
        all_data_y += tmp_data_y
        all_colors_idx += [int(idx) for i in range(len(tmp_data_y))]

    if ax:
        #plt.sca(ax)
        sc = ax.scatter(all_data_x, all_data_y, c=all_colors_idx, cmap=cmap, s=s, alpha=alpha)
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.)
            cbar = fig.colorbar(sc, cax=cax)
            cbar.ax.set_ylabel(cbar_label, labelpad=11)
            cbar.ax.locator_params(nbins=4)
    
    else:
        sc = plt.scatter(all_data_x, all_data_y, c=all_colors_idx,  cmap=cmap, s=s, alpha=alpha)
        if colorbar:
            plt.colorbar(sc)
    return project_data

def fitBaselineSpam(model, data=None, basedata=None, ticalag=25, ticadim=4, ticaunits='ns', factor=1):
    import pandas as pd
    from htmd.metricdata import MetricData
    import numpy as np

    all_data = pd.DataFrame(columns=["epoch", "sim", "x", "y"])

    if basedata:
        spam_model = spamming(model)
        for idx, data in enumerate(spam_model):
            if idx == 0:
                projectdata, basetica = fitBaseline(data, basedata, ticalag=ticalag, 
                                                    ticadim=ticadim, ticaunits=ticaunits,tica=True)
            else:
                projectdata = basetica.tic.transform(data)
            
            for i in projectdata:
                all_data = all_data.append({"epoch":idx, "x":i[0], "y":i[1]}, ignore_index=True)
    else:
        from htmd.adaptive.adaptive import epochSimIndexes
        epoch_idx = epochSimIndexes(model.data.simlist)

        if isinstance(data, np.ndarray):
            simids = np.array([i.simid for i in model.data.simlist])
            first_frames = np.array(np.ceil(model.data.trajLengths/factor), dtype=int)
            spam_data = {i:[data[np.sum(first_frames[0:i])]] for i in simids} #List within list for compatibility
            choosen_frames = [np.sum(first_frames[0:i]) for i in simids]
        elif isinstance(data, MetricData):
            spam_data = data.dat
        else:
            spam_data = model.data.dat
        
        for key, val in epoch_idx.items():
            for sim in val:
                x, y = spam_data[sim][0][0:2]
                all_data = all_data.append({"epoch":key, "sim":sim, "x":x, "y":y}, ignore_index=True)
    
    return all_data

def fitBaseline(data, basedata, ticalag=25, ticadim=4, ticaunits='frames', tica=False):
    from htmd.projections.tica import TICA
    from htmd.model import Model
    basetica = TICA(basedata, ticalag, units=ticaunits)
    basetica.tic.set_params(dim = ticadim)

    if isinstance(data, Model):
        data = data.data.parent
    
    if len(data.dat.shape) != 2:
        data = np.concatenate(data.dat)
    
    #Fix to avoid memory errors
    try:
        projectdata = basetica.tic.transform(data)
    except:
        projectdata = []
        chunks = np.linspace(0, len(data), 10, dtype=int)
        _ = [projectdata.extend(basetica.tic.transform(data[i:chunks[idx+1]])) for idx, i in enumerate(chunks[0:-1])]
        projectdata = np.array(projectdata)
    
    if tica:
        return projectdata, basetica    
    return projectdata

def fitBaselineWithMetrics(projected_simlist, base_simlist, metric, ticalag=25, ticadim=4, ticaunits='frames', tica=False):
    from htmd.projections.tica import TICA
    from htmd.projections.metric import Metric
    from htmd.model import Model

    """
    Implement a MetricB that returns the TICA tranformation of a MetricA
    1) Calculate MetricA for each trajectory
    2) TICA transform based on a basetica (basetica.tic.t)
    """

    basetica_metric = Metric(base_simlist)
    basetica_metric.set(metric)
    basetica = TICA(basetica_metric, ticalag, units=ticaunits)
    basetica.tic.set_params(dim = ticadim)

    def metricToTica(mol, metric, tica):
        metric_data = metric.project(mol)
        return tica.tic.transform(metric_data)  

    tica_metric = Metric(projected_simlist)
    tica_metric.set((metricToTica, (metric, basetica)))
    projectdata = tica_metric.project().dat

    if tica:
        return projectdata, basetica    
    return projectdata

def plot_model_by_rmsd(model, rmsd_dat=None, rmsd_mean=None, rmsd_std=None, cmap='jet', legend=True, save=None, ax=None):
    import matplotlib as mpl

    if rmsd_dat is None and rmsd_mean is None and rmsd_std is None:
        raise RuntimeError("Either rmsd_dat or rmsd_mean & rmsd_std should be defined")
    
    if rmsd_dat:
        rmsd_mean = getStateStatistic(model, rmsd_dat, states=range(model.micronum), statetype="micro", method=np.mean)
        # rmsd_min = getStateStatistic(model, rmsd_dat, states=range(model.micronum), statetype="micro", method=np.min)
        rmsd_std = getStateStatistic(model, rmsd_dat, states=range(model.micronum), statetype="micro", method=np.std)
    
    rmsd_mean = np.array(rmsd_mean).ravel()
    # rmsd_min = np.array(rmsd_min).ravel()
    rmsd_std = np.array(rmsd_std).ravel()
    
    plt.sca(ax) if ax else plt.figure(figsize=(8,8))

    cmap = mpl.cm.get_cmap(cmap, model.macronum)
    c = [cmap(model.macro_ofmicro[i]) for i in range(model.micronum)]

    macro_pop = np.round(model.eqDistribution(plot=False)*100, 1)
    macro_sort = np.argsort(macro_pop)[::-1]
    macros = range(model.macronum)

    for idx, i in enumerate(macro_sort):
        c = cmap(len(macros) - 1 - idx)
        macro_in_micro = np.where(model.macro_ofmicro == i)[0]
        tmp_x = rmsd_mean[macro_in_micro]
        tmp_y = rmsd_std[macro_in_micro]
        sc = plt.scatter(tmp_x, tmp_y, color=c, s=15, alpha=0.5, edgecolor=c, label=f"Macro {i}, {macro_pop[i]}%")
    if legend:
        plt.legend()
    plt.xlabel("Mean RMSD \n by microstate (Å)")
    plt.ylabel(r"$\it{SD\ RMSD \ by\ microstate\ (Å)}$")
    _ = plt.ylim(0, np.max(rmsd_std)*1.2)
    _ = plt.xlim(0, np.max(rmsd_mean)*1.2)

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)
    return rmsd_mean, rmsd_std
   

def plot_model_by(model, dat1, dat2, method1=np.mean, s=15, method2=np.mean, cmap="Set1", legend=True, ylabel=None, xlabel=None, ylim=(None, None), xlim=(None, None)):
    import matplotlib as mpl

    if method1:
        cum_dat1 = np.array(getStateStatistic(model, dat1, states=range(model.micronum), statetype="micro", method=method1)).ravel()
    else:
        cum_dat1 = dat1

    if method2:
        cum_dat2 = np.array(getStateStatistic(model, dat2, states=range(model.micronum), statetype="micro", method=method2)).ravel()
    else:
        cum_dat2 = dat2

    cmap = mpl.cm.get_cmap(cmap, model.macronum)
    c = [cmap(model.macro_ofmicro[i]) for i in range(model.micronum)]

    macro_pop = np.round(model.eqDistribution(plot=False)*100, 1)

    for i in range(model.macronum):
        c = cmap(i)
        macro_in_micro = np.where(model.macro_ofmicro == i)[0]
        tmp_x = cum_dat1[macro_in_micro]
        tmp_y = cum_dat2[macro_in_micro]
        sc = plt.scatter(tmp_x, tmp_y, color=c, s=s, alpha=0.6, edgecolor=c, label=f"Macro {i}, {macro_pop[i]}%")
    
    if legend: plt.legend() 
    
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    xmin, xmax = xlim
    if xmin is None:
        xmin = np.min(cum_dat1)*0.8
    if xmax is None:
        xmax = np.max(cum_dat1)*1.2
    
    ymin, ymax = ylim
    if ymin is None:
        ymin = np.min(cum_dat2)*0.8
    if ymax is None:
        ymax = np.max(cum_dat2)*1.2

    _ = plt.ylim(ymin, ymax)
    _ = plt.xlim(xmin, xmax)
    return cum_dat1, cum_dat2


def numClusters(model):
    """ Heuristic that calculates number of clusters from number of frames """
    import numpy as np
    numFrames = model.data.numFrames
    K = int(max(np.round(0.6 * np.log10(numFrames / 1000) * 1000 + 50), 100))  # heuristic
    if K > numFrames / 3:  # Ugly patch for low-data regimes ...
        K = int(numFrames / 3)
    return K
