from htmd.ui import *

def model_analysis_all_plots(model, out_folder):
	from IDP_model import plot_contacts, plot_dihedral, contact_plot, contact_plot_by_atom
	from htmd.ui import MetricDistance, MetricDihedral, Molecule
	from IDP_analysis import aux_plot
	import numpy as np
	#Additional plots
	mol = Molecule(model.data.simlist[0].molfile)
	cont_metric = MetricDistance(sel1="noh and protein", sel2="noh and resname MOL", 
	                           groupsel1="residue", groupsel2="all", 
	                           threshold=5, metric="contacts")
	aux_plot(model, cont_metric, mol, plot_contacts, np.mean,
		save=out_folder + "/3_contacts.png")

	dih_metric = MetricDihedral(protsel="protein")
	aux_plot(model, dih_metric, mol, plot_dihedral, np.std,
		save=out_folder + "/4_dihedral.png", chain_id="P1",
		start_index=54)

	contact_map_metric = MetricDistance(sel1="noh and protein", sel2="noh and resname MOL", 
	                           groupsel1="residue", threshold=5, metric="contacts")
	mapping = contact_map_metric.getMapping(mol)
	aux_plot(model, contact_map_metric, mol, contact_plot_by_atom, np.mean,
		mapping=mapping, save=out_folder + "/5_cm.png")


	all_contact_metric = MetricDistance(sel1="noh and protein or noh and resname MOL", 
	                           sel2="noh and protein or noh and resname MOL", 
	                           groupsel1="residue", groupsel2="residue", threshold=4, metric="contacts")
	mapping = all_contact_metric.getMapping(mol)
	aux_plot(model, all_contact_metric, mol, contact_plot, np.mean,
		mapping=mapping, cols=2, rows=model.macronum/2, plot=False, save=out_folder + "/6_full_cm.png")

def get_params_model(model):
    return {'clusters' : len(model.micro_ofcluster),
    'lag' : int(round(model.lag*model.data.fstep, 0)), 
    'macroN' : model.macronum
  }

def save_structures(model, outdir, states, numsamples, statetype, **kwargs):
    import os
    from glob import glob
    if len(states) != len(numsamples) and len(numsamples) != len(statetype):
        print("Length of states, numsamples and statetype should match")
        return
    os.makedirs(outdir, exist_ok=True)
    for idx, i in enumerate(states):
        m = model.getStates(statetype=statetype[idx], states=[i], numsamples=int(numsamples[idx]), **kwargs)
        for struct in m:
            for frame in range(struct.numFrames):
                out_name = "{}/{}_{}_{}.pdb".format(outdir, statetype[idx], i , frame)
                struct.frame = frame
                struct.resname[struct.resname=='HSD'] = "HIS"
                struct.write(out_name, sel="not name CAY CY OY NT CAT")
    return glob(outdir+"/*pdb")

def metastable_states(model):
    import numpy as np
    metastable_sets = []
    for i in range(model.macronum):
        metastable_sets.append(np.where(model.macro_ofmicro == i)[0])
    model.metastable_sets = np.array(metastable_sets)

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