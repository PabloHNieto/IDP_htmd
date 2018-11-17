from htmd.ui import *
from copy import deepcopy
from htmd.util import ensurelist
from htmd.builder.builder import sequenceID
from matplotlib import pylab as plt
from matplotlib import cm as colormaps
import matplotlib.pyplot as plt
import matplotlib as mpl


def contactVecToMatrix(vector, atomIndexes):
    Molecule()
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

def contact_plot(ver, axes, mapping, mol, title):
    global mpbl

    three_letter ={'V':'VAL', 'I':'ILE', 'L':'LEU', 'E':'GLU', 'Q':'GLN', \
    'D':'ASP', 'N':'ASN', 'H':'HSD', 'W':'TRP', 'F':'PHE', 'Y':'TYR',    \
    'R':'ARG', 'K':'LYS', 'S':'SER', 'T':'THR', 'M':'MET', 'A':'ALA',    \
    'G':'GLY', 'P':'PRO', 'C':'CYS', 'MOL':'MOL'}

    vector = np.ones(ver.shape, dtype=bool)
    colors= ver
    truecontacts = np.zeros(len(vector), dtype=bool)
    
    # Creating the 2D contact maps
    cm, newmapping, uqAtomGroups = contactVecToMatrix(vector, mapping.atomIndexes)
    num = len(uqAtomGroups)
    cmtrue, _, _ = contactVecToMatrix(truecontacts, mapping.atomIndexes)
    im = axes.imshow(cmtrue / 2, interpolation='none', vmin=0, vmax=1, aspect='equal',
               cmap='Greys')  # /2 to convert to gray from black
    
    rows, cols = np.where(cm)
     
    if isinstance(colors, np.ndarray) and isinstance(colors[0], float):
        mpbl = colormaps.ScalarMappable(cmap=colormaps.jet)
        mpbl.set_array(colors)
        colors = mpbl.to_rgba(colors)
    if len(colors) == len(vector):
            colors = colors[newmapping[rows, cols]]
    
    rows = rows + 0.5
    cols = cols + 0.5
    
    all_res = [three_letter[i]+str(b) for i, b in zip(['MOL'] + list(mol.sequence()['P1']) , list(set(mol.resid)))]
    all_res = all_res[1:] + all_res[0:1]
#    all_res = [three_letter[i]+str(b) for i, b in zip(list(mol.sequence()['P1']), list(set(mol.resid))) if i!='G'] 
    axes.set_xticklabels(all_res, rotation="vertical", ha='left')
    axes.set_yticklabels(all_res, va='bottom')
    axes.set_xticks(np.arange(0, num, 1))
    axes.set_yticks(np.arange(0, num, 1))
#    for tick in axes.get_xticklabels():
#        tick.set_rotation(90)
#    plt.setp(axes.xaxis.get_majorticklabels(), rotation=90)
##    
    axes.scatter(rows, cols,  c=colors, lw=0)
#    axes.scatter(rows, cols, lw=0)
    axes.set_axisbelow(True)
    axes.set_title(title)
#    axes.xaxis.set_ticks_position('both')
#    axes.yaxis.set_ticks_position('both')
#    axes.xaxis.labelpad = 0.5
#    axes.minorticks_on()
    axes.grid(which="both", color='#969696', linestyle='-', linewidth=1)
    axes.tick_params(axis='both', which='both', length=0)
    axes.set_xlim([0, num ])
    axes.set_ylim([0, num ])


def plotContactMap(model, data, states=None, statetype='macro', save=None):
    """ Plots the standard deviation of dihedral angles by macrostates

    Parameters
    ----------
    data : []
    Array with the positions of each node. Calculated by default setting the most populated
    node in the center and the rest with a distance to the center proporcional to their connectivity degree.
    start_index : int
    The number of the first residue of the molecule. For renumbering purposes
    chain_id : str
    The chain ID of the molecule to select in order to create labels
    save : str
    Path of the file in which to save the figure

    Examples
    --------
    >>> IDPmodel = IDPModel(model)
    >>> IDPmodel.modelToNetwork(min_rate=0.05)
    >>> print(IDP.Gc)
    """
    mean = getStateStatistic(model, data, states=states,
        statetype=statetype, method=np.mean, weighted=False)

    mol = Molecule(model.data.parent.simlist[0].molfile)

    mapping = MetricDistance(sel1='noh and protein or noh and resname MOL', sel2='noh and protein or noh and resname MOL',
                             groupsel1="residue", groupsel2="residue", metric="contacts").getMapping(mol)

    f, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 12), dpi=200)
    for b,r in enumerate(axes.flat):
        contact_plot(mean[b], r, mapping, mol, "Macro {}".format(b))
    if save:
        plt.savefig(save)
    a=input()

if __name__ == '__main__':
    
    #Load contact Data
    contact_data = MetricData()
    contact_data.load("/workspace6/phn_sj403/18-9-2017_O43806_wild_0_sj403/contact_data.dat")
    print("Data Loaded")
    
    #Load Model
    model = Model()
    model.load("/workspace6/phn_sj403/18-9-2017_O43806_wild_0_sj403/model_19_09_2018_contacts_4_8_300Micro.dat")
    print("Model Loaded")

    plotContactMap(model, contact_data, states=range(model.macronum), save="/home/pablo/contact_test.png")
