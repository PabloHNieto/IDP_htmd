from htmd.ui import *
from IDP_htmd.IDP_analysis import *
from IDP_htmd.IDP_model import *
# from IDP_htmd.model_utils import *
# from IDP_htmd.MetricRadiusGyration import MetricRG
import matplotlib.pyplot as plt



class ModelAnalysis(object):
    """Pipeline for the analysis of MD data usin MSM using HTMD package
    
    Attributes
    ----------
    bulk_split : TYPE
        Metric to create a new macrostate using model.utils.bulk_split
    cluster : int
        Number of clusters to create the MSM
    cluster_scan : int[] or None
        Arrays of clusters to test using mode.utils.cluster_scan
    fes : bool
        Wether to plot or not the free energy surface with the two fist TICA dimensions. It does not work if ticadim is None
    input_folder : str
        Folder with the adaptive run data 
    macronum : int
        Number of macronum to create the model
    metrics : TYPE
        Metrics to project the raw MD data.
    model : None | str | htmd.Model.model
        It will create, load or assing a htmd.model.Model class
    modellag : int
        Lag to create the model
    modelunits : str
        Units for the lag time. Can be 'ns' or 'frames'
    mol : TYPE
        Description
    out_folder : str
        Output folder to store results from the analysis
    plot_contacts : tuple
        Array of tuples indicating the output name, VMD selection and treshold to measure contacts
    plot_dihedral : str | None
        Name for the plot of standard deviation of dihedrals
    plot_mol_contacts : str | None
        Base name for the plot between protein and MOL
    rg_analysis : bool
        Whether to perfom an analysis of the radious of gyration of the model.
    save_model : bool
        Wethter to save the model or not.
    skip : int
        Number of frames to skip while projecting the data
    start_index : int
        Startign index for the molecule of your simulations. To create right labels for plotting
    ticadim : int | None
        Number of TICA dimension to retrieve
    ticalag : int
        lag to be used for tica.
    """
    
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.out_folder = output_folder
        self.metrics = None
        self.skip = 1
        self.ticadim = 3
        self.ticalag = 20
        self.cluster = 500
        self.macronum = 5
        self.modellag = 20
        self.modelunits = "ns"
        self.fes = True
        self.save_model = True
        self.rg_analysis = True
        self.plot_dihedral = 'dihedral'
        self.plot_contacts = None
        self.plot_mol_contacts = False
        self.start_index = 1
        self.cluster_scan = False
        self.model = None
        self.bulk_split = None
        self.start_index = 0
        self.kinetics = False
        self.temperature = 310
        self.concentration = None

        if (not self.out_folder or not self.input_folder):
            print("Not data or out folder provided")

    def perfom_analysis(self):
        """

        
        Returns
        -------
        TYPE
            Description
        """
        import os 
        from htmd.molecule.molecule import Molecule
        from IDP_htmd.model_utils import scan_clusters

        # Checking essential paramenters
        if (not self.metrics):
            print("Metrics have not been set")
            return 
        os.makedirs(self.out_folder, exist_ok=True)

        self.handle_model()

        self.additional_plots()

        if self.cluster_scan:
            scan_clusters(self.model, self.cluster_scan, self.out_folder)

        if self.kinetics:
            self.calc_kinetics()
        self.generate_html_summary()

    def write_parameters(self, excluded=['out_folder', 'input_folder', 'model', 'mol', 'plot_contacts', 'plot_dihedral',
        'fes', 'plot_mol_contacts', 'rg_analysis', 'start_index', 'bulk_split', 'save_model']):
        """Write the parameters set for the analysis to a json file
        
        Parameters
        ----------
        excluded : list, optional
            List of excluded parameters not be included in the output file
        """
        import json
        # Writing model parms to file

        json_model = self.__dict__.copy()
        json_model['metrics'] = { "{}-{}".format(idx, str(met.__class__).split("'")[-2].split(".")[-1]): met.__dict__ for idx, met in enumerate(self.metrics)}

        for key in excluded:
            json_model.pop(key, None)
        json_info = {'model': json_model}

        with open(self.out_folder+'file.txt', 'w') as file:
          file.write(json.dumps(json_info))

    def handle_model(self):
        """Creates a model is model is not set. Loads a model from a string. Or assign a model to self.model.out_folder

        Calling this function results in self.model to be and htmd.model.Model class
        """
        from htmd.model import Model
        from htmd.molecule.molecule import Molecule

        if not self.model:
            from IDP_htmd.IDP_analysis import analyze_folder
            print("Creating new analysis")
            self.write_parameters()
            self.model = analyze_folder(self.input_folder, self.out_folder, self.skip, self.metrics, self.cluster,
                self.ticadim, self.ticalag, self.modellag, self.modelunits, self.macronum, self.bulk_split, 
                self.fes, self.rg_analysis, self.save_model)

        if isinstance(self.model, str):
          try:
            print("Loading model")
            model = Model()
            model.load(self.model)
            self.model = model
          except:
            print("Could not load the model")
            return

        if isinstance(self.model, Model):
            print("Model loaded")
            pass

        self.mol = Molecule(self.model.data.simlist[0].molfile)


    def additional_plots(self):
        """Createa additinal plot following the parameters set for the instance of the class
        """
        if self.plot_dihedral:
            self.plot_dih(self.plot_dihedral)

        if self.plot_mol_contacts:
            self.plot_mol_contact()
            self.plot_atom_mol_contact()

        if self.plot_contacts:
            for name, sel, threshold in self.plot_contacts:
                self.plot_contact_map(name, sel, threshold)

    def plot_contact_map(self, name, selection="noh and protein", threshold=4):
        """Generate a residue-residue contact plot by macrostate given a VMD selection
        
        Parameters
        ----------
        name : str
            Output name for the generated plot
        selection : str, optional
            VMD selection to create a matrix contact plot
        threshold : int, optional
            Threshold distance in angstrom to discrinate contact vs no-contact
        """
        contact_metric = MetricDistance(sel1=selection, sel2=selection,
            groupsel1="residue", groupsel2="residue", threshold=threshold, metric="contacts")
        
        mapping = contact_metric.getMapping(self.mol)
        aux_plot(self.model, contact_metric, self.mol, contact_plot, skip=self.skip, method=np.mean,
          mapping=mapping, cols=2, rows=int(self.model.macronum/2)+self.model.macronum%2,
          plot=False, save=self.out_folder + "/{}.png".format(name))

    def plot_dih(self):
        """Creates a plot of the standard deviation of the dihedral angles of the protein by macrostate
        """
        dih_metric = MetricDihedral(protsel="protein")
        aux_plot(self.model, dih_metric, self.mol, plot_dihedral, skip=self.skip, method=np.std,
            save=self.out_folder + "/{}.png".format(self.plot_dihedral), chain_id="P1",
            start_index=self.start_index)

    def plot_atom_mol_contact(self, sel1="noh and protein", sel2="noh and resname MOL", threshold=5):
        """Plot a molecule-residue contact map.
        
        Parameters
        ----------
        sel1 : str, optional
            VMD selection. Grouped by residue.
        sel2 : str, optional
            VMD selection
        threshold : int, optional
            Threshold distance in angstrom to discrinate contact vs no-contact
        """
        label = ['M{}-{}%'.format(i, np.round(percent*100, 2)) for i, percent in enumerate(self.model.eqDistribution(plot=False))]
        mol_contact_map_metric = MetricDistance(sel1=sel1, sel2=sel2,
            groupsel1="residue", threshold=threshold, metric="contacts")
        mapping = mol_contact_map_metric.getMapping(self.mol)

        aux_plot(self.model, mol_contact_map_metric, self.mol, contact_plot_by_atom, skip=self.skip, method=np.mean,
            mapping=mapping, label=label, save=f'{self.out_folder}/{self.plot_mol_contacts}_by_atom.png')

    def plot_mol_contact(self, sel1="noh and protein", sel2="noh and resname MOL", threshold=4):
        """Plot a molecule-residue contact map.
        
        Parameters
        ----------
        sel1 : str, optional
            VMD selection. Grouped by residue.
        sel2 : str, optional
            VMD selection
        threshold : int, optional
            Threshold distance in angstrom to discrinate contact vs no-contact
        """
        labels = generate_labels(self.mol)
        mol_contact_map_metric = MetricDistance(sel1=sel1, sel2=sel2, 
            groupsel1="residue", groupsel2="all", threshold=5, metric="contacts")

        mapping = mol_contact_map_metric.getMapping(self.mol)
        aux_plot(self.model, mol_contact_map_metric, self.mol, plot_contacts, skip=self.skip, method=np.mean,
            mod=self.model, title="Contacts by residue",
            plot=False, save=f'{self.out_folder}/{self.plot_mol_contacts}.png')

    def generate_html_summary(self):
        """Generates a html report with all the data generated
        """
        from IDP_htmd.jinja.render import Render
        import json

        date = self.out_folder.split("/")[-2]
        pictures = glob(f"{self.out_folder}/*png")
        try:
            with open(f"{self.out_folder}/file.txt", "r") as myfile:
                js = json.load(myfile)
        except Exception as e:
            print(e)
            js = None

        if len(pictures)>0:
            info = {
                'date': date,
                'pictures': pictures,
                'folder': self.out_folder }

            if js:
                info['metrics'] = js['model']['metrics']
                js['model'].pop('metrics', None)
                info['parameters'] = js

            r = Render("analysis", f"{self.out_folder}/IDP_summary", info)

    def calc_kinetics(self):
        """Calculates kinetics rates for the model
        
        Returns
        -------
        pandas.DataFrame
            Dataframe containing value for mfpton, mfptoff, kon, koff, kdeq and g0eq
            from a source macro to each other macrostate
        """
        import pandas as pd
        
        columns = ['mfpton', 'mfptoff', 'kon', 'koff', 'kdeq', 'g0eq']
        kin_summary = pd.DataFrame(columns=columns)
        source = self.model.macronum - 1 if self.bulk_split else None
        for  i in range(self.model.macronum):
            kin_rates = Kinetics(self.model, self.temperature, 
                concentration=self.concentration, source=source, sink=i).getRates()
            source = kin.source
            row = [ kin[i] for i in columns ]
            kin_summary.append(row, ignore_index=True)
        kin_summary.to_csv(f'{self.out_folder}kin.csv')

        return kin_summary

if __name__ == '__main__':
    from htmd.projections.metricdistance import MetricDistance
    from htmd.model import Model

    mt = ModelAnalysis("/workspace8/excitome/adaptiveRun/O75376_MOR_58/",
        "/home/pablo/testModel/")

    mt.metrics = [
                MetricDistance(
                sel1="noh and protein",
                sel2="noh and protein",
                metric="contacts",
                threshold=5,
                groupsel1="residue",
                groupsel2="residue")
            ]

    # mt = Model()
    # mt.load("/home/pablo/testModel/model.dat")
    # mt.model = model
    # mt.model = "/home/pablo/testModel/model.dat"
    # mt.plot_dihedral = "2_dihedral"
    mt.macronum = 4
    mt.plot_contacts = [
        ('all_contacts', 'noh and protein', 5),
        ('backbone', 'noh and backbone', 5),
        ('sidechain', 'noh and sidechain', 4),
    ]
    mt.write_parameters()
    mt.generate_html_summary()

    # mt.perfom_analysis()