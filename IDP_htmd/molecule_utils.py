#!/shared/pablo/miniconda3/bin/python
from htmd.molecule.molecule import Molecule
import numpy as np 

def create_dummy_atom():
    return Molecule("/shared/pablo/IDP_htmd/IDP_htmd/ref_files/dummy.pdb")

def _read_header_pocket(filename):
    starting_coor = []
    vector_coor = np.array([0, 0, 0], dtype=np.float)
    cube_dim = np.array([0, 0, 0])
    _coor = 0
    with open(filename, 'r') as my_file:
        for idx, l in enumerate(my_file.readlines()):
            if idx == 2:
                starting_coor = np.fromstring(l, sep=" ")[1:]
            elif idx > 2:
                l = np.fromstring(l, sep=" ")
                if len(l) == 4:
                    vector_coor[_coor] = l[_coor + 1]
                    cube_dim[_coor] = l[0]
                    _coor += 1
                else:
                    break
    return starting_coor, vector_coor, cube_dim

def _read_body_pocket(filename):
    BORH_ANGSTROM_RATIO = 0.529177
    AR_THRESHOLD = 0.1

    # Reading header
    starting_coor, vector_coor, cube_dim = _read_header_pocket(filename)
    x_vec, y_vec, z_vec = vector_coor
    x_dim, y_dim, z_dim = cube_dim
    x_start, y_start, z_start = starting_coor

    # Handling body
    body = np.genfromtxt(filename, dtype='float32', skip_header=7)
    body = body.ravel().reshape((x_dim, y_dim, z_dim)) * BORH_ANGSTROM_RATIO


    coords = []
    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                if body[x][y][z] >= AR_THRESHOLD:
                    coor = np.array([[round(x * x_vec + x_start, 3)],
                            [round(y * y_vec + y_start, 3)],
                            [round( z * z_vec + z_start, 3)]])
                    coords.append(coor * BORH_ANGSTROM_RATIO) 

    return np.round(np.array(coords, dtype='float32'), 3), body[body>AR_THRESHOLD]

def read_pocket(filename):
    """Read pocket from .cube files
    
    Loads the volumetric information of a .cube file representing the
    pocket shape and location

    Parameters
    ----------
    filename : str
        Path to the location of a .cube file
    """
    pocket_mol = create_dummy_atom()

    pocket_mol.coords, occ = _read_body_pocket(filename)

    numAtoms = pocket_mol.coords.shape[0]
    pocket_mol.altloc = pocket_mol.altloc.repeat(numAtoms)
    pocket_mol.atomtype = pocket_mol.atomtype.repeat(numAtoms)
    pocket_mol.beta = pocket_mol.beta.repeat(numAtoms)
    pocket_mol.chain = pocket_mol.chain.repeat(numAtoms)
    pocket_mol.charge = pocket_mol.charge.repeat(numAtoms)
    pocket_mol.element = pocket_mol.element.repeat(numAtoms)
    pocket_mol.insertion = pocket_mol.insertion.repeat(numAtoms)
    pocket_mol.name = pocket_mol.name.repeat(numAtoms)
    pocket_mol.masses = pocket_mol.masses.repeat(numAtoms)
    pocket_mol.occupancy = occ
    pocket_mol.resid = np.arange(1, numAtoms + 1)
    pocket_mol.resname = pocket_mol.resname.repeat(numAtoms)
    pocket_mol.segid = pocket_mol.segid.repeat(numAtoms)
    pocket_mol.record = pocket_mol.record.repeat(numAtoms)
    pocket_mol.serial = np.arange(1, numAtoms + 1)
    return pocket_mol

def view_pocket(mol):
    pass

if __name__ == "__main__":
    from IDP_htmd.molecule_utils import create_dummy_atom
    d = create_dummy_atom()
    mol = Molecule("//shared/pablo/IDP_htmd/IDP_htmd/ref_files/cube_files/macro_0_1.pdb")
    mol.view(sel="noh and protein", color="Name", style="Licorice")
    pocket = read_pocket("/shared/pablo/IDP_htmd/IDP_htmd/ref_files/cube_files/output.cube")
    pocket.write("/home/pablo/test.pdb")
    # pocket.view(style="CPK", color="Occupancy")