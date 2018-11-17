
def buildProtein(sequence, outdir, start_idx=1):
    from htmd.builder.preparation import proteinPrepare
    from htmd.ui import Molecule, amber
    import sys
    sys.path.append("/home/pablo/PyRosetta4.Release.python36.linux.release-197")
    import pyrosetta
    pyrosetta.init()
    pose = pyrosetta.pose_from_sequence(sequence, 'fa_standard')
    pose.dump_pdb('/tmp/test.pdb')
    mol = Molecule('/tmp/test.pdb')
    mol = proteinPrepare(mol)
    mol.resid = mol.resid + start_idx
    mol.set('segid', 'P1')
    mol.filter('backbone')
    _ = amber.build(mol, outdir=outdir, ionize=False
        # , caps=None
        )
    return outdir+'/structure.prmtop'

def unfoldProtein(prmtop, inputpdb, outpdb):
    import sys
    sys.path.append("/workspace6/Folding/Unfolder/scripts/")
    from openmm import unfoldOpenMM
    import parmed
    struct = parmed.load_file(prmtop)
    system = struct.createSystem()
    unfoldOpenMM(inputpdb, outpdb,
            100000, 800000, system)

def build_seq(sim_seq, folder):
    prmtop = buildProtein(sim_seq, folder)
    unfoldProtein(prmtop, folder+"/structure.pdb",
        folder+'/structure.pdb')

def main():
    import sys
    import pandas as pd 
    out_folder = sys.argv[2] + "/"
    sim_seq = pd.read_csv(sys.argv[1])
    for mor_id, _, sim_seq in sim_seq.values:
        print(mor_id)
        build_seq(sim_seq, out_folder)

if __name__ == '__main__':
    main()

