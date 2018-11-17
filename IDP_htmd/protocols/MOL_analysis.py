from htmd.ui import *
from IDP_htmd.IDP_analysis import *
from IDP_htmd.IDP_model import *
from IDP_htmd.model_utils import scan_clusters
import matplotlib.pyplot as plt
plt.switch_backend('agg')

#Analysis
def make_analysis(out_folder=None, data_folder=None, model=None, clu=None):
    import json

    if (not out_folder or not data_folder):
        print("Not data or out folder provided")
        return

    # Writing model parms to file
    json_model = model.copy()
    json_model['metrics'] = { "{}-{}".format(idx, met.__class__): met.__dict__ for idx, met in enumerate(model['metrics'])}
    json_info = {'model': json_model, "clu":clu}

    with open(out_folder+'file.txt', 'w') as file:
        file.write(json.dumps(json_info))

    print(data_folder)

    if isinstance(model, dict):
        model = analyze_folder(data_folder, out_folder, **model)
        print("Creating new analysis")

    if isinstance(model, str):
        try:
            model = Model()
            model.load(model)
        except:
            print("Could not load the model")
            return

    mol = Molecule(model.data.simlist[0].molfile)

    #Scan cluster effect on ITS
    if clu:
        scan_clusters(model, clu, out_dir=out_folder)
        model = Model()
        model.load(out_folder+"model.dat")
    # Create bulk state
    idpmol_contact_metric = MetricDistance(sel1="noh and protein",
                                           sel2="noh and resname MOL",
                                           groupsel1="all", groupsel2="all", threshold=4, metric="contacts")
    any_contact = get_data(model, idpmol_contact_metric)
    contacts = np.array(getStateStatistic(model, any_contact, states=range(model.micronum), statetype="micro"))
    min_contacts = np.where(contacts == np.min(contacts))[0]
    model.createState(min_contacts)



    #Additional plots
    cont_metric = MetricDistance(sel1="noh and protein", sel2="noh and resname MOL",
                                 groupsel1="residue", groupsel2="all",
                                 threshold=5, metric="contacts")

    aux_plot(model, cont_metric, mol, plot_contacts, np.mean, mod=model,
             save=out_folder + "/3_contacts.png")

    dih_metric = MetricDihedral(protsel="protein")
    aux_plot(model, dih_metric, mol, plot_dihedral, np.std,
             save=out_folder + "/4_dihedral.png", chain_id="P1",
             start_index=54)

    label = ['M{}-{}%'.format(i, np.round(percent*100, 2)) for i, percent in enumerate(model.eqDistribution(plot=False))]
    contact_map_metric = MetricDistance(sel1="noh and protein", sel2="noh and resname MOL",
                                        groupsel1="residue", threshold=5, metric="contacts")
    mapping = contact_map_metric.getMapping(mol)
    aux_plot(model, contact_map_metric, mol, contact_plot_by_atom, np.mean,
             mapping=mapping, label=label, save=out_folder + "/5_cm.png")

    labels = generate_labels(mol, 'MOL')
    print(labels)
    all_contact_metric = MetricDistance(sel1="noh and protein or noh and resname MOL",
                                        sel2="noh and protein or noh and resname MOL",
                                        groupsel1="residue", groupsel2="residue", threshold=4, metric="contacts")
    mapping = all_contact_metric.getMapping(mol)
    aux_plot(model, all_contact_metric, mol, contact_plot, np.mean, ligand=True,
             mapping=mapping, cols=2, rows=int(model.macronum/2)+model.macronum%2,
             xlabels=labels, ylabels=labels,
             plot=False, save=out_folder + "/6_full_cm.png")


    labels = [lab for lab in labels if "GLY" not in lab ]
    sidechain_contact_metric = MetricDistance(sel1="noh and sidechain or noh and resname MOL",
                                              sel2="noh and sidechain or noh and resname MOL",
                                              groupsel1="residue", groupsel2="residue", threshold=4, metric="contacts")
    sc_mapping = sidechain_contact_metric.getMapping(mol)
    aux_plot(model, sidechain_contact_metric, mol, contact_plot, np.mean,
             mapping=sc_mapping, cols=2, rows=int(model.macronum/2)+model.macronum%2,
             xlabels=labels, ylabels=labels,
             plot=False, save=out_folder + "/7_sidechain_cm.png")

    labels = generate_labels(mol, 'MOL')
    backbone_contact_metric = MetricDistance(sel1="noh and backbone or noh and resname MOL",
                                             sel2="noh and backbone or noh and resname MOL",
                                             groupsel1="residue", groupsel2="residue", threshold=4, metric="contacts")
    bb_mapping = backbone_contact_metric.getMapping(mol)
    aux_plot(model, backbone_contact_metric, mol, contact_plot, np.mean,
             mapping=bb_mapping, cols=2, rows=int(model.macronum/2)+model.macronum%2,
             xlabels=labels, ylabels=labels,
             plot=False, save=out_folder + "/8_backbone_cm.png")

