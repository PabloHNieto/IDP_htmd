
def cis_angle(mol):
    import numpy as np
    from htmd.ui import Dihedral, MetricDihedral
    dih = np.array(Dihedral.proteinDihedrals(mol, dih=('omega')))
    met = MetricDihedral(dih, sincos=False)
    return np.array(met.project(mol))

if __name__ == '__main__':
    from htmd.ui import Molecule
    base_folder = '/shared/pablo/PHN_htmd/PHN_htmd/ref_files/MD_trajectory/'
    mol = Molecule(base_folder + 'structure.pdb')
    mol.read(base_folder + 'structure.psf')
    mol.read(base_folder + 'output.xtc')
    mol.filter("protein")
    dih = cis_angle(mol)