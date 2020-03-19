import numpy as np

def analyze_folder(folder=None, out_folder="/tmp",  skip=1, metrics=None, clu=500, tica=True, ticadim=5, 
    tica_lag=20, model_lag=10, model_units='ns', macro_N=10, bulk_split=False, fes=True, rg_analysis=True, save=True, data_fstep=None): 
    """Analysis script for create a Markov State Model
    
    Creates and returns a Markov State Model given a data folder.
    Intented to follow up the evolution of an adaptive sampling run.
    Allows to save the model ans several informative plots
    
    Parameters
    ----------
    folder : str
        Data folder where adaptive is running
    out_folder : str
        Output folder to store derived data
    skip : int
        Number of frames to skip while projecting the MD data
    metrics : [:class: `Metric` object]
        Metric array used to project the data
    clu : int
        Number of cluster to create using the MiniBatchKMeans method.
    tica: bool
        Wether to use TICA of GWPCA for dimensionality reduction
    ticadim : int
        Number of TICA dimension to project the data. If None, the model will be created using the raw projected data
    tica_lag : int, optional
        Description
    model_lag : int
        Number of ns used to create the model
    model_units : str, optional
        Description
    macro_N : int
        Number of macrostate to split the final Markov State Model
    fes : bool, optional
        If true it will save a plot projecting the first two TICA dimension. Requires ticadim to be defined
    rg_analysis : bool, optional
        If true, a plot with information relative to the radious of gyration of the molecule will be created.
    save : bool, optional
        If true, the model will be saved in the outputs folder
    
    Returns
    -------
    :class:`Model`
        Final model
    """
    from htmd.model import Model
    from htmd.molecule.molecule import Molecule
    from htmd.simlist import simlist
    from htmd.projections.metric import Metric 
    from sklearn.cluster import MiniBatchKMeans
    from IDP_htmd.IDP_model import plot_RG 
    from IDP_htmd.model_utils import create_bulk
    from glob import glob
    import os
    
    try:
        os.mkdir(out_folder)
    except:
        print("Folder already exists")

    try:
        fsims = np.load(f"{folder}/simlist.npy", allow_pickle=True)
        print(f"Loaded {folder}/simlist.npy")
    except:
        print("Creating simlist")
        sims = glob(folder + 'filtered/*/')
        fsims = simlist(sims, folder+'filtered/filtered.pdb')
    metr = Metric(fsims, skip=skip)
    metr.set(metrics)
    
    #Check if this gives problems to ITS

    try:
        model = Model(file=f"{out_folder}/model.dat")
        out_data = model.data
        print(f"Loading model: {out_folder}/model.dat")
    except:
        if tica and ticadim:
            from htmd.projections.tica import TICA
            print("Projecting TICA")
            tica = TICA(metr, tica_lag)
            out_data = tica.project(ticadim)
        elif not tica and ticadim:
            from htmd.projections.gwpca import GWPCA
            data = metr.project()
            data.dropTraj()
            print("using GWPCA")
            gwpca = GWPCA(data, tica_lag)
            out_data = gwpca.project(ticadim)
        else:
            print("Not using TICA")
            data = metr.project()
            data.dropTraj()
            out_data = data

    #Avoid some possibles error while clustering
    if data_fstep: out_data.fstep = data_fstep
    x = True
    while x:
        try:
            out_data.cluster(MiniBatchKMeans(n_clusters=clu), mergesmall=5)
            x = False
        except Exception as e:
            raise Exception("Error " + str(e))

    model = Model(out_data)
    model.plotTimescales(plot=False, save=f"{out_folder}/1_its.png")

    if macro_N:
        model.markovModel(model_lag, macro_N, units=model_units)
        
        if bulk_split:
            try:
                print("Starting bulk splitting")
                create_bulk(model, bulk_split)
            except Exception as e:
                print("Could not perform the bulk splitting")
                print(e)
                
        model.eqDistribution(plot=False, save=f"{out_folder}/1.2_eqDistribution.png")

        if rg_analysis:
            from IDP_htmd.IDP_analysis import rg_analysis
            mol = Molecule(model.data.simlist[0].molfile)
            rg_data = rg_analysis(model, skip=skip)
            plot_RG(rg_data, mol,  save=f"{out_folder}/1.4_rg.png")

        # if fes and ticadim:
            # model.plotFES(0, 1, temperature=310, states=True,
            #     plot=False, save=f"{out_folder}/1.3_fes.png")

    if save:
        model.save(f"{out_folder}/model.dat")

    return model


def rg_analysis(model, **kwargs):
    from htmd.model import getStateStatistic
    from IDP_htmd.MetricRadiusGyration import metricRG
    from IDP_htmd.model_utils import get_data
    import numpy as np

    rg_data = get_data(model, metricRG, **kwargs)
    rg_mean = getStateStatistic(model, rg_data, states=range(model.macronum))
    rg_std = getStateStatistic(model, rg_data, states=range(model.macronum), method=np.std)
    
    aggregate_dat = []
    for i in rg_data.dat:
      aggregate_dat += i.ravel().tolist()
    aggregate_dat = np.array(aggregate_dat)

    rg_mean = np.append(rg_mean, np.mean(aggregate_dat))
    rg_std = np.append(rg_std, np.std(aggregate_dat))

    return np.array([ rg_mean, rg_std ])



