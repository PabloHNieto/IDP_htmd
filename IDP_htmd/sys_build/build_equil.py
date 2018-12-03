from htmd.ui import *
from htmd.molecule.util import maxDistance

# def write_equil_unfold(outdir_build, outdir_equil, 
#     steps=7500000, temperature=500):
#     from htmd.protocols.equilibration_v2 import Equilibration
#     print(outdir_equil)
#     # High temperature simulation for unfolding
#     md = Equilibration()
#     md.runtime = steps
#     md.timeunits = 'steps'
#     md.temperature = temperature
#     md.nvtsteps = steps
#     md.constraintsteps = -1
#     md.acemd.dielectric = '80'
#     try:
#         md.write(outdir_build, outdir_equil)
#     except:
#         print ("---> ", outdir_build)

def write_equil(outdir_build, outdir_equil, unfold=False,
                steps=7500000, temperature=310):
    from htmd.protocols.equilibration_v2 import Equilibration
    print(outdir_equil)
    md = Equilibration()
    md.runtime = steps
    md.timeunits = 'steps'
    md.nvtsteps = steps
    md.temperature = temperature
    md.constraintsteps = -1
    if unfold:
        md.acemd.dielectric = '80'
    try:
        md.write(outdir_build, outdir_equil)
    except:
        print ("---> ", outdir_build)

def send2simulate(equil):
    mdx=SlurmQueue()
    mdx.partition = 'multiscale'
    mdx.exclude = ['giallo', 'green']
    mdx.submit(equil)

def write_production(input_folder, output, temperature=310, steps='10000000'):
    from htmd.protocols.production_v6 import Production
    md = Production()
    md.temperature = temperature
    md.acemd.bincoordinates = 'output.coor'
    md.acemd.extendedsystem  = 'output.xsc'
    md.acemd.binvelocities=None
    md.acemd.binindex=None
    md.runtime=steps
    md.timestep="steps"
    md.adaptive=True
    try:
        md.write(input_folder, output)
    except:
        print ("---> ", input_folder)

def build_worker_charm22star(start_mol, idx, out="/tmp", saltconc=0.015):
    from random import randint
    topos22 = ['top/top_all22star_prot.rtf',
            'top/top_water_ions.rtf']
    params22 = ['par/par_all22star_prot.prm', 
            'par/par_water_ions.prm']
    outdir = "{}{}_eq".format(out, idx)
    print(outdir)
    D = 41
    keep = True
    while keep:
        mol = start_mol.copy()
        if mol.numFrames > 1:
            x =  randint(50, start_mol.numFrames-1)
            mol.dropFrames(keep=[x])
        mol = proteinPrepare(mol)
        mol.set("segid", "P1")
        mol.filter(sel="protein")
        mol.center()
        mol_move = maxDistance(mol, 'all')
        if mol_move < D:
            keep = False
    smol = solvate(mol, minmax=[[-D, -D, -D], [D, D, D]])
    molbuilt = charmm.build(smol, outdir = outdir, 
        caps=None, topo=topos22, param=params22, saltconc=saltconc)
    return outdir

def parallel_build(protein, outdir, iterations=1):
    import multiprocessing as mp
    jobs = []
    for idx in range(1, iterations+1):
        work = mp.Process(target=build_worker_charm22star,
                          args=(protein, idx, outdir))
        jobs.append(work)
        work.start()

def parallel_writing(func, build, equil):
    import multiprocessing as mp
    jobs=[]
    for b, e in zip(build, equil):
        work = mp.Process(target=func,
            args=(b, e))
        jobs.append(work)
        work.start()

def main_B():
    idps = glob("/workspace8/excitome/0_IDP/*/")
    for idp in idps:
        mol = Molecule(idp+'structure.pdb')
        mol.dropFrames(keep=[-1])
        parallel_build(mol, idp.replace("0_IDP", "1_unfold"),
            iterations=1)

def main_E():
    # builds = glob("/workspace8/excitome/1_unfold/*/*/")
    for i in builds:
        try:
            write_equil_unfold(i, i.replace("1_unfold", "2_equilU"))
        except:
            print("-->", i)

def main_R():
    # equils = glob("/workspace8/excitome/2_equilU/*/*/")
    equils = [
        "/workspace8/excitome/2_equilU/Q16665_MOR_5/1_eq/",
        "/workspace8/excitome/2_equilU/P04637_MOR_1/1_eq/",
        "/workspace8/excitome/2_equilU/P46100_MOR_29/1_eq/",
        ]
    send2simulate(equils)

if __name__ == '__main__':
    # main_B()
    # main_E()
    main_R()
