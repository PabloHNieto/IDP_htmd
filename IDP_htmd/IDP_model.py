from htmd.model import Model 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  
from tqdm import tqdm
import networkx as nx
from htmd.util import ensurelist

three_letter ={'V':'VAL', 'I':'ILE', 'L':'LEU', 'E':'GLU', 'Q':'GLN', \
                'D':'ASP', 'N':'ASN', 'H':'HSD', 'W':'TRP', 'F':'PHE', 'Y':'TYR', \
                'R':'ARG', 'K':'LYS', 'S':'SER', 'T':'THR', 'M':'MET', 'A':'ALA', \
                'G':'GLY', 'P':'PRO', 'C':'CYS', 'MOL':'MOL'}

def plot_RG(data, mol, chain_id="P1", labels=None, plot=True, save=None):
    """Plots the radious of gyration of a model by macrosate
    
    Parameters
    ----------
    data : np.ndarray
        Radious of gyration data for the model
    mol : <htmd.molecule.molecule.Molecule>
        Molecule object of the model
    chain_id : str, optional
        Chain id of the mol obejct, used to define the labels (the default is "P1")
    labels : [string], optional
        Array with the labels of the x axis 
        (the default is None, which results in the computatiosn of the default labels)
    plot : bool, optional
        Wether to plot the picture (the default is True, which plots the pictures)
    save : string, optional
        File path where to save the picture (the default is None, which does not save the picture)
    
    """
    from IDP_htmd.MetricRadiusGyration import predictRG

    plt.figure(figsize=(7,7))

    seq = mol.sequence()[chain_id]
    lower_bound, upper_bound = predictRG(len(seq))
    limits = [ [lower_bound, upper_bound] for i in range(len(data[0])) ]

    if not labels:
        labels = [ 'Macro-{}'.format(i) for i,_ in enumerate(data[0])]
        labels[-1] = "Aggregated"
    
    plt.bar(range(len(data[0])), data[0], yerr=data[1], width=0.8)
    plt.xticks(range(len(data[0])), labels, rotation=45)

    for idx,(lower_bound, upper_bound) in enumerate(limits):
        plt.hlines(lower_bound, idx+0.1, idx-0.1, colors='red', linestyles='solid', label='Folded-RG')
        plt.hlines(upper_bound, idx+0.1, idx-0.1, colors='blue', linestyles='solid', label='Coil-RG')

    if plot:
        plt.show()

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)

def plot_contacts(data, mol, idx=None, model=None, labels=None, 
    title=None, save=None, vmin=0, vmax=1, 
    xlabel=None, cmap="viridis", ax=None):
    """[summary]
    
    Parameters
    ----------
    data : np.ndarray
        Radious of gyration data for the model
    mol : <htmd.molecule.molecule.Molecule>
        Molecule object of the model
    labels : [string], optional
        [description] (the default is None, which will create default labels for the y axis)
    idx : int, optional
        Starting index to be  (the default is None, which will infeer it from the molecule)
    model : <htmd.model.Model>, optional
        Model object, used to infeer the y labels 
        (the default is None, which will set up an enumeration of the y labels)
    title : string, optional
        Title of the picture (the default is None, which does not includes any title)
    plot : bool, optional
        Wether to plot the picture (the default is True, which plots the pictures)
    save : string, optional
        File path where to save the picture (the default is None, which does not save the picture)
    vmin : int, optional
        Minimun value to be shown in the contact map 
        (the default is 0, which [default_description])
    vmax : int, optional
        Maximun value to be shown in the contact map 
        (the default is 1, which [default_description])
    xlabel : [string], optional
        Arrays of labels to be used in the x axes 
        (the default is None, which will create deafult one with residue name and residue number)
    cmap : str, optional
        Matplotlib colormap name (the default is "viridis")
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if not idx:
        idx = mol.resid[0]
    xlabels = [f'{three_letter[b]}{i+idx}' for i,b in enumerate(mol.sequence()['P1'])]

    if ax == None: ax = plt.gca() 
    else: ax = ax

    im = ax.imshow(np.flip(data, 0)[::-1], vmin=vmin, vmax=vmax, cmap=cmap)

    if title:
        plt.title(title)

    ax.set_xticklabels(xlabels, rotation='vertical', ha="center")
    ax.tick_params(length=0)
    ax.set_xticks(range(len(xlabels)))

    if not labels:
        labels = ['Macro-{}'.format(i) for i in range(len(data))]

    if model:
        ylabels = ['Macro {}: {:5.2f}%'.format(idx, i*100) for idx, i in enumerate(model.eqDistribution(plot=False),0 )]
        y = [i for i in range(model.macronum)]
        ax.set_yticks(y)
        ax.set_yticklabels(ylabels)
    else:
        # y = [i-0.5 for i in range(len(labels))]
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, ha="right")
    ax.set_ylim(len(labels)-0.5, -0.5)
    ax.set_xlabel(xlabel)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r"$P(contact)$", rotation=90)
    plt.sca(ax)

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)


def _atom_contact_plot(ver, ax, mol, label, cmap="viridis"):

    im = ax.imshow(ver, vmin=0, vmax=1, aspect="equal", cmap=cmap)
    all_res = [three_letter[i]+str(b) for i, b in zip(list(mol.sequence()['P1']), list(set(mol.resid))[1:])] 

    ax.tick_params(length=0)
    ax.set_yticks([len(ver)*0.2])
    ax.set_yticklabels([label], rotation="vertical")
    ax.set_xticks([i+0.3 for i in range(len(list(mol.sequence()['P1'])))])
    ax.set_xticklabels(all_res, rotation="vertical", ha='center')
    ax.set_aspect('auto')
    return im

def contact_plot_by_atom(all_data, mol, mapping=None, label=None, chain_id="P1", title=None, save=None, ax=None, **kwds):
    """Plots a contact map for each macrostate 
    
    Parameters
    ----------
    all_data : np.ndarray
        Array with the data to be plotted with dimenasions number of micros/macros x shape of mapping
    mol : <htmd.molecule.molecule.Molecule>
        Molecule used to extract the sequences.
    mapping : [type], optional
        Mapping object of the data used to create the data
    label : [string], optional
        Array with the labels of the x axis 
        (the default is None, which results in the computatiosn of the default labels)
    chain_id : str, optional
        Chain id of the mol obejct, used to define the labels (the default is "P1")
    title : string, optional
        Title of the picture (the default is None, which does not includes any title)
    plot : bool, optional
        Wether to plot the picture (the default is True, which plots the pictures)
    save : string, optional
        File path where to save the picture (the default is None, which does not save the picture)
    """
    sequence_length = len(mol.sequence()[chain_id])
    dimension = int(len(all_data[0])/sequence_length)
    
    if ax: plt.sca(ax)
    
    f, axes = plt.subplots(ncols=1, 
        nrows=len(all_data), 
        figsize=(10, len(all_data)*1.5), 
        sharex=True)
    if title:
        f.suptitle(title)
    f.subplots_adjust(hspace=0)

    if not label:
        label = [ "Macro-{}".format(i) for i in range(len(all_data))]

    if len(all_data) == 1:
            data = np.transpose(all_data[0].reshape(sequence_length, dimension))
            z = _atom_contact_plot(data, axes, mol, label[0], **kwds)
    else:
        for b,(r, data) in enumerate(zip(axes.flat, all_data)):
            data = np.transpose(data.reshape(sequence_length, dimension))
            z = _atom_contact_plot(data, r, mol, label[b], **kwds)

    cbar_ax = f.add_axes([0.92, 0.15, 0.025, 0.7])
    cbar = f.colorbar(z, cax=cbar_ax)
    cbar.set_label(r"$P(contact)$", rotation=90)


    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.sca(axes[-1])

def plot_dihedral(data, mol, start_index=1, chain_id="X", save=None, factor=1, title=None, ylabel=None, return_data=False, **kwds):
    """ Plots the standard deviation of dihedral angles by macrostates

    Parameters
    ----------
    data : [] 
    start_index : int
            The number of the first residue of the molecule. For renumbering purposes
    chain_id : str
            The chain ID of the molecule to select in order to create labels
    save : str
            Path of the file in which to save the figure
            
    """
    import matplotlib as mpl
    mpl.rcParams['axes.linewidth'] = 0.8
    def _merge(arr):
        half1 = arr[0::2]
        half2 = arr[1::2]
        newArr = [ x for i in zip(half1[1:], half2[:-1]) for x in i ]
        return np.array([half1[0]] + newArr + [half2[-1]])
    
    label = [f"{three_letter[b]}{i + start_index}" for i,b in enumerate(mol.sequence()[chain_id])]
    
    # newLabel = ["{}-{}".format(label[idx], label[idx + 1]) for idx in list(range(len(label)-1))]
    fig, axes = plt.subplots(figsize=(factor*12, 3*factor*len(data)), 
        nrows=len(data), ncols=1, sharex=True, dpi=300)

    if title: fig.suptitle(title)

    try:
        len(axes)
    except:
        axes = [axes]
    
    all_sin = []
    all_cos = []

    for idx, (ax, dat)  in enumerate(zip(axes, data)):
        newSTDsin = _merge(dat[0::2])
        newSTDcos = _merge(-1*dat[1::2])
        all_sin.append(newSTDsin)
        all_cos.append(newSTDcos)
        xticks = [i - 0.51 for i in range(0, len(newSTDsin)+1, 2)]

        ax.bar(range(len(newSTDsin)), newSTDsin, lw=0.5)
        ax.bar(range(len(newSTDsin)), newSTDcos, lw=0.5)

        ax.axhline(y=0.50, xmin=0, xmax=len(dat), linestyle="dashed", color="red", lw=factor)
        ax.axhline(y=0.15, xmin=0, xmax=len(dat), linestyle="dashed", color="black", lw=factor)
        # ax.axhline(y=1, xmin=0, xmax=len(dat), linestyle="dashed", color="red", lw=0.25)

        ax.axhline(y=-0.50, xmin=0, xmax=len(dat), linestyle="dashed", color="red", lw=factor)
        ax.axhline(y=-0.15, xmin=0, xmax=len(dat), linestyle="dashed", color="black", lw=factor)

        ax.axhline(y=0.0, xmin=0, xmax=len(dat), color="black", lw=0.5)
        
        # ax.axhline(y=-1, xmin=0, xmax=len(dat), linestyle="dashed", color="red", lw=0.25)
        
        ax.set_xlim(-1, len(newSTDsin))
        ax.set_ylim(-1, 1)
        ax.xaxis.grid()

        # ax.yaxis.set_label_position("right")
        if ylabel and len(ylabel) >= len(axes): ax.set_ylabel(ylabel[idx], rotation=90)
        ax.yaxis.set_tick_params(width=.5)
        ax.set_yticks([0.75, -0.25])
        # , rotation='vertical', ha="center",
        ax.set_yticklabels(["Sine", "- Cosine"], ha="right", rotation='vertical')
        ax.set_xticks(xticks)
        ax.set_xticklabels(label, rotation='vertical', ha="left")
            
        fig.subplots_adjust(hspace=0.0)

    ax.xaxis.set_tick_params(width=.5)
    
    plt.sca(axes[0])
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    if return_data: 
        return np.array(all_cos), np.array(all_sin), fig
    
    return fig
    

def plot_helicity(data, yerr=None, beta_helix=False, label=None, xticks=None, ylabel=None, cmap="jet", legend=True, ax=None, save=None):
    import matplotlib as mpl
    data_length = len(data)
    data_shape = len(data[0])
    colors = mpl.cm.get_cmap(cmap, data_length)
    c = np.array(colors(range(data_length)))
    w = 0.9/len(data)
   
    label = [f"Dataset {i}" for i in range(data_length)] if not label else label
    
    plt.sca(ax) if ax else None        

    for idx, i in enumerate(data):
        tmp_yerr = [0 for i in range(data_shape)] if yerr is None else yerr[idx]
        if beta_helix and len(data) > 2:
            raise RuntimeError("Length of the data should be maximum 2 if plot beta and helix values")
        elif beta_helix:
            plt.errorbar(np.arange(0, data_shape), i, yerr=tmp_yerr, label=label[idx], lw=3, color=c[idx], fmt="-o", capsize=3)
        else: 
            plt.bar(np.arange(0, data_shape) + w*idx, i, width=w,
                    alpha=0.5, color=c[idx], label=label[idx], capsize=1, edgecolor = 'black')
            plt.errorbar(np.arange(0, data_shape) + w*idx, i, yerr=tmp_yerr, lw=1, c='k', linestyle="", capsize=3)

    plt.ylim(0, 1.5)
    plt.xlim(-1, data_shape)

    if xticks:
        _ = plt.xticks(np.arange(data_shape) + w, xticks, rotation=90, ha="center")
    plt.legend() if legend else None 

    plt.ylabel(ylabel) if ylabel else plt.ylabel("Helicity (%)")

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)



def _contactVecToMatrix(vector, atomIndexes):
    from copy import deepcopy
    # Calculating the unique atom groups in the mapping
    uqAtomGroups = []
    atomIndexes = deepcopy(list(atomIndexes))
    for ax in atomIndexes:
        ax[0] = ensurelist(ax[0])
        ax[1] = ensurelist(ax[1])
        if ax[0] not in uqAtomGroups:
            uqAtomGroups.append(ax[0])
        if ax[1] not in uqAtomGroups:
            uqAtomGroups.append(ax[1])
    uqAtomGroups.sort(key=lambda x: x[0])  # Sort by first atom in each atom list
    num = len(uqAtomGroups)

    matrix = np.zeros((num, num), dtype=vector.dtype)
    mapping = np.ones((num, num), dtype=int) * -1
    for i in range(len(vector)):
        row = uqAtomGroups.index(atomIndexes[i][0])
        col = uqAtomGroups.index(atomIndexes[i][1])
        matrix[row, col] = vector[i]
        matrix[col, row] = vector[i]
        mapping[row, col] = i
        mapping[col, row] = i
    return matrix, mapping, uqAtomGroups

def _contact_plot(ver, axes,  mol, mapping, dpi=300, title=None, xlabels=None, ylabels=None,  legend=False, labels=True, s=110, start_idx=0):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import cm as clmap
    plt.rcParams.update({'font.size': 15})
    global mpbl
    vector = np.ones(ver.shape, dtype=bool)
    colors = ver
    truecontacts = np.zeros(len(vector), dtype=bool)
    
    # Creating the 2D contact maps
    cm, newmapping, uqAtomGroups = _contactVecToMatrix(vector, mapping.atomIndexes)
    num = len(uqAtomGroups)
    _ , _, uq = _contactVecToMatrix(truecontacts, mapping.atomIndexes)
    
    if not xlabels:
        xlabels = [mol.resname[atom[0]] + str(mol.resid[atom[0]] + start_idx) for atom in uq]

    if not ylabels:
        ylabels = [mol.resname[atom[0]] + str(mol.resid[atom[0]] + start_idx) for atom in uq]

    rows, cols = np.where(cm)

    if isinstance(colors, np.ndarray) and isinstance(colors[0], float):
        mpbl = clmap.ScalarMappable(cmap=clmap.jet)
        mpbl.set_array(colors)
        colors = mpbl.to_rgba(colors)
    if len(colors) == len(vector):
        colors = colors[newmapping[rows, cols]]
    
    rows = rows + 0.5
    cols = cols + 0.5

    if labels:
        axes.set_xticklabels(xlabels, rotation="vertical", ha='left')
        axes.set_yticklabels(ylabels, va='bottom')
        axes.set_xticks(np.arange(0, num, 1))
        axes.set_yticks(np.arange(0, num, 1))

    axes.scatter(rows, cols, s=s, c=colors, lw=0, marker="s")
    axes.set_axisbelow(True)
    axes.set_title(title)
    # axes.grid(which="both", color='#969696', linestyle='-', linewidth=1)
    axes.tick_params(axis='both', which='both', length=0)
    axes.set_xlim([0, num])
    axes.set_ylim([0, num])

    # if legend:
    #     divider = make_axes_locatable(axes)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     plt.colorbar(im, cax=cax)


def contact_plot(data, mol, cols, rows, model=None, factor=10, plot=True, save=None, **kwargs):
    """Plots a contact matrix for each data in the array.
    Plot an NxM figure where N corresponds to rows and M to cols.
    All data component should belong to the same type of molecule
    
    Parameters
    ----------
    data : list
        List of arrays with the data to be plotted for each subplot
    mol : htmd.molecule.Molecule
        Molecule object. Used to create labels.
    cols : int
        Number of cols for subplots
    rows : int
        Number of rows for subplots
    model : htmd.model.Model, optional
        Model where the data is extracted. Used for labeling only.
    plot : bool, optional
        Whether to plot or not the figure
    save : str, optional
        Path to save the picture created
    **kwargs
        Additional arguments for plotting such a title or axes labels for each individual plot
    """
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(cols*factor, rows*factor),
        dpi=300)

    axes = axes.flatten() if type(axes) == np.ndarray else [axes]

    for idx, (dat, ax) in enumerate(zip(data, axes)):
        _contact_plot(ver=np.array(dat), axes=ax, title=f'Macro-{idx + 1}', mol=mol, dpi=300, **kwargs)
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)
    
    if plot:
        plt.show()
    return fig

def generate_labels(mol, *args):

    all_res = [three_letter[i]+str(b) for i, b in zip(list(mol.sequence()['P1']), list(set(mol.resid[mol.atomselect('protein')])))]

    _ = [ all_res.append(i.upper()) for i in args ]
    return all_res

def generate_labels2(mol, mapping, *args):
    labels = []
    for atoms, _ in mapping.atomIndexes:
        label = mol.resname[atoms[0]] + str(mol.resid[atoms[0]])
        if not label in labels:
            labels.append(label)  
    return labels

def plot_mfpt(data, cmap="tab20c", save=None):
    """Plots the logarithm of the mfpt
    
    Parameters
    ----------
    data : np.array
        Matrix with the mpft between states
    save : string, optional
        Path of the file in which to save the figure (the default is None, which does not save the plot)
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.title("MFPT between macros")
    ax = plt.gca()
    im = plt.imshow(np.log10(data), cmap=cmap)
    # ax2.colorbar()
    #
    # _ = ax2.set_xticks(range(len(sorted_model_labels)))
    # _ = ax2.set_xticklabels(sorted_model_labels2, rotation='vertical')
    #
    # _ = ax2.set_yticks(range(len(sorted_model_labels)))
    # _ = ax2.set_yticklabels(sorted_model_labels2)
    ax.set_ylabel("From ...")
    ax.set_xlabel("To ...")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    cbar = plt.colorbar(im, cax=cax)
    # $\it{Italics}$
    cbar.set_label("$\it{log_{10} (mfpt)}$")
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)


def plot_goal_rmsd(rmsd_data, std=None, colors=None, plot=True, save=None):
    plt.title("Min RMSD by epoch")
    plt.xlabel("Epoch")
    plt.ylabel("RMSD (A)")
    plt.ylim((0, np.max(rmsd_data)+2))
    
    if not colors:
        color_tree = plt.cm.get_cmap("viridis", len(rmsd_data))
        colors = [color_tree(i) for i in range(len(rmsd_data))]

    for idx, (col, data) in enumerate(zip(colors, rmsd_data)):
        plt.plot(range(len(data)), data, color=col)
        plt.plot(range(len(data)), data, "o", color=col)
        if std is not None:
            plt.fill_between(range(len(data)), data+std[idx], data-std[idx], alpha=0.5, facecolor=col)
    
    if plot:
        plt.show()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)

def plot_animated_contacts(contacts, ylab=None, xlab=None, save=None):
    import matplotlib.animation as animation
    fig, ax=plt.subplots(figsize=(17, 10))
    container = []

    for _, row in contacts.iterrows():
        frame_data = np.transpose(row.contacts.reshape((87, -1)))
        title = ax.text(0.5,1.05,"Epoch {}".format(row.epoch), 
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes, )
        im = ax.imshow(frame_data, vmin=0, vmax=3)
        if xlab:
            ax.set_xticks(np.arange(len(xlab)) + 0.0)
            ax.set_xticklabels(xlab, rotation=90)
        if ylab:
            ax.set_yticks(np.arange(len(ylab)) + 0.0)
            ax.set_yticklabels(ylab)
        container.append([im, title])
        
    ani = animation.ArtistAnimation(fig, container, interval=200, blit=True)
    if save:
        ani.save(save, writer='imagemagick', fps=2)
    return ani

if __name__ == "__main__":
    from htmd.model import getStateStatistic
    from htmd.projections.metric import MetricData
    from htmd.projections.metricdistance import MetricDistance
    from htmd.model import Model
    from htmd.molecule.molecule import Molecule
    import numpy as np

    data = MetricData()
    data.load("/workspace8/p27_sj403/10-11-2018_p27_short_sj403/analysis/17_11_2018/testing.dat")
    model = Model()
    model.load("/workspace8/p27_sj403/10-11-2018_p27_short_sj403/analysis/17_11_2018/model.dat")
    mol = Molecule(model.data.simlist[0].molfile)
    mean_dat = getStateStatistic(model, data, range(model.macronum))
    met = MetricDistance(sel1="noh and protein or resname MOL", sel2="noh and protein or resname MOL",
        groupsel1="residue", groupsel2="residue", metric="distances", pbc=False)
    mapping = met.getMapping(mol)
    contact_plot(mean_dat, mol, rows=2, cols=2, model=model, plot=False, save="/home/pablo/test.png", mapping=mapping)
    
