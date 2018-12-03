from htmd.ui import *

def get_params_model(model):
    return {'clusters' : len(model.micro_ofcluster),
    'lag' : int(round(model.lag*model.data.fstep, 0)), 
    'macroN' : model.macronum
  }

def save_structures(model, outdir, states, numsamples, statetype, 
    modifications=None, **kwargs):
    import os
    from glob import glob
    if len(states) != len(numsamples) and len(numsamples) != len(statetype):
        print("Length of states, numsamples and statetype should match")
        return
    os.makedirs(outdir, exist_ok=True)
    for idx, i in enumerate(states):
        m = model.getStates(statetype=statetype[idx], states=[i], numsamples=int(numsamples[idx]), **kwargs)
        for struct in m:
            if modifications:
                for prop, setting, sel in modifications:
                    struct.set(prop, setting, sel)
            for frame in range(struct.numFrames):
                out_name = "{}/{}_{}_{}.pdb".format(outdir, statetype[idx], i , frame)
                struct.frame = frame
                struct.resname[struct.resname=='HSD'] = "HIS"
                struct.write(out_name, sel="not name CAY CY OY NT CAT")
    return glob(outdir+"/*pdb")

def get_weighted(model, total_struct):
    import numpy as np
    print("here")
    population = model.eqDistribution(plot=False)
    out_structs = np.array([int(total_struct*pop) for pop in population ])

    idx_max = np.where(population == np.max(population))
    if np.sum(out_structs) < total_struct:
        out_structs[idx_max] += total_struct - np.sum(out_structs)
    elif np.sum(out_structs) > total_struct:
        out_structs[idx_max] -= total_struct - np.sum(out_structs)
    return out_structs

def metastable_states(model):
    import numpy as np
    metastable_sets = []
    for i in range(model.macronum):
        metastable_sets.append(np.where(model.macro_ofmicro == i)[0])
    model.metastable_sets = np.array(metastable_sets)

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

def create_bulk(model, metric):
    from IDP_htmd.IDP_model import get_data
    data = get_data(model, metric)
    data_by_micro = np.array(getStateStatistic(model, data, states=range(model.micronum), statetype="micro"))
    min_contacts = np.where(data_by_micro == np.min(data_by_micro))[0]
    model.createState(min_contacts)
    metastable_states(model)
    print(min_contacts)

def cluster_macro(model, data, macro, method=np.mean):
    from sklearn.cluster import AffinityPropagation, MiniBatchKMeans, DBSCAN, Birch
    from IDP_htmd.IDP_model import plot_contacts
    mol = Molecule(model.data.simlist[0].molfile)
    data_by_micro = getStateStatistic(model, data, states=model.metastable_sets[macro], statetype="micro", method=method)
    clusters = AffinityPropagation().fit(data_by_micro)
    # clusters = Birch().fit(data_by_micro)
    for i in range(len(clusters.cluster_centers_)):
        label_micro = model.metastable_sets[macro][np.where(clusters.labels_ == i)[0]]
        print(i, label_micro)
        cluster = getStateStatistic(model, data,
                                 states=label_micro, statetype="micro", method=np.mean)
        labels = [ 'Micro {}'.format(i) for i in label_micro ]
        plot_contacts(cluster, mol, labels=labels)
        # model.createState(label_micro)

def compute_all_mfpt(model):
    all_mfpt = []
    for source in range(model.macronum):
        all_mfpt.append([model.msm.mfpt(source, sink) for sink in range(model.macronum)])
    return np.array(all_mfpt)

def scan_clusters(model, nclusters, out_dir):
    for i in nclusters:
        model.data.cluster(MiniBatchKMeans(n_clusters=i), mergesmall=5)
        new_mod = Model(model.data)
        new_mod.plotTimescales(plot=False, save=out_dir+"1_its-{}_clu".format(i))