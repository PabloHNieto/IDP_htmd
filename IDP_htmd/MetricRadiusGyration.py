
import logging

from moleculekit.projections.projection import Projection

import numpy as np

logger = logging.getLogger(__name__)


class MetricRG(Projection):
    _atomic_mass = dict(H=1.01, He=4.00, Li=6.94, Be=9.01, B=10.81, C=12.01,
                        N=14.01, O=16.00, F=19.00, Ne=20.18, Na=22.99, Mg=24.31,
                        Al=26.98, Si=28.09, P=30.97, S=32.07, Cl=35.45, Ar=39.95,
                        K=39.10, Ca=40.08, Sc=44.96, Ti=47.87, V=50.94, Cr=52.00,
                        Mn=54.94, Fe=55.85, Co=58.93, Ni=58.69, Cu=63.55, Zn=65.39,
                        Ga=69.72, Ge=72.61, As=74.92, Se=78.96, Br=79.90, Kr=83.80,
                        Rb=85.47, Sr=87.62, Y=88.91, Zr=91.22, Nb=92.91, Mo=95.94,
                        Tc=98.00, Ru=101.07, Rh=102.91, Pd=106.42, Ag=107.87,
                        Cd=112.41, In=114.82, Sn=118.71, Sb=121.76, Te=127.60,
                        I=126.90, Xe=131.29, Cs=132.91, Ba=137.33, La=138.91,
                        Ce=140.12, Pr=140.91, Nd=144.24, Pm=145.00, Sm=150.36,
                        Eu=151.96, Gd=157.25, Tb=158.93, Dy=162.50, Ho=164.93,
                        Er=167.26, Tm=168.93, Yb=173.04, Lu=174.97, Hf=178.49,
                        Ta=180.95, W=183.84, Re=186.21, Os=190.23, Ir=192.22,
                        Pt=195.08, Au=196.97, Hg=200.59, Tl=204.38, Pb=207.2,
                        Bi=208.98, Po=209.00, At=210.00, Rn=222.00, Fr=223.00,
                        Ra=226.00, Ac=227.00, Th=232.04, Pa=231.04, U=238.03,
                        Np=237.00, Pu=244.00, Am=243.00, Cm=247.00, Bk=247.00,
                        Cf=251.00, Es=252.00, Fm=257.00, Md=258.00, No=259.00,
                        Lr=262.00, Rf=261.00, Db=262.00, Sg=266.00, Bh=264.00,
                        Hs=269.00, Mt=268.00)

    def __init__(self, sel='protein', ndim=1):
        self.sel = sel
        self.mass = None
        self.t_mass = None
        self._ndim = ndim

    def _precalculate(self, mol):
        self.mass, self.t_mass = self._calcarrays(mol)

    def _calcarrays(self, mol):
        mol = mol.copy()
        mol.filter(self.sel, _logger=False)

        mass = np.array([self._atomic_mass[i] for i in mol.element])
        tmass = np.sum(mass)

        return mass, tmass

    def _calculateMolProp():
        pass


    @staticmethod
    def predict(aa):
        """Source: http://www.scfbio-iitd.res.in/software/proteomics/rg.jsp
        According to Wilkings et at. parameters
        Source:
        'Hydrodynamic radii of native and denatured proteins measured by pulse
        field gradient NMR techniques'
        Wilkins DK, Grimshaw SB, Receveur V, Dobson CM, Jones JA, Smith LJ

        Parameters
        ----------
        aa : int
          Number of aminoacids

        Returns
        -------
        tuple
          Lower and upper theoretical limits for the radious of gyration
          for a protein of the given length
        """
        upper_limit = 2.54 * aa**0.522
        lower_limit = (3 / 5)**0.5 * 4.75 * aa**0.29
        return lower_limit, upper_limit

    def project(self, mol):
        """ Project molecule.

        Parameters
        ----------
        mol : :class:`Molecule <htmd.molecule.molecule.Molecule>`
            A molecule object to project

        Returns
        -------
        data : np.ndarray
            An array containing the projected data.
        """

        mol = mol.copy()
        mol.filter(self.sel, _logger=False)
        center_of_mass = np.sum(mol.coords * self.mass[:, None, None], axis=0) / self.t_mass
        dist_from_com = np.sqrt(np.sum((mol.coords - center_of_mass)**2, axis=1))
        rg = np.sqrt(np.sum(self.mass[: , None] * dist_from_com**2, axis=0) / self.t_mass)
        return rg

    def getMapping(self, mol):
        pass

def predictRG(aa):
    """Source: http://www.scfbio-iitd.res.in/software/proteomics/rg.jsp
    According to Wilkings et at. parameters
    Source:
    'Hydrodynamic radii of native and denatured proteins measured by pulse
    field gradient NMR techniques'
    Wilkins DK, Grimshaw SB, Receveur V, Dobson CM, Jones JA, Smith LJ

    Parameters
    ----------
    aa : int
        Number of aminoacids

    Returns
    -------
    tuple
        Lower and upper theoretical limits for the radious of gyration
        for a protein of the given length
    """
    upper_limit = 2.54 * aa**0.522
    lower_limit = (3 / 5)**0.5 * 4.75 * aa**0.29
    return lower_limit, upper_limit

def metricRG(mol, sel="protein"):
    _atomic_mass = dict(H=1.01, He=4.00, Li=6.94, Be=9.01, B=10.81, C=12.01,
                        N=14.01, O=16.00, F=19.00, Ne=20.18, Na=22.99, Mg=24.31,
                        Al=26.98, Si=28.09, P=30.97, S=32.07, Cl=35.45, Ar=39.95,
                        K=39.10, Ca=40.08, Sc=44.96, Ti=47.87, V=50.94, Cr=52.00,
                        Mn=54.94, Fe=55.85, Co=58.93, Ni=58.69, Cu=63.55, Zn=65.39,
                        Ga=69.72, Ge=72.61, As=74.92, Se=78.96, Br=79.90, Kr=83.80,
                        Rb=85.47, Sr=87.62, Y=88.91, Zr=91.22, Nb=92.91, Mo=95.94,
                        Tc=98.00, Ru=101.07, Rh=102.91, Pd=106.42, Ag=107.87,
                        Cd=112.41, In=114.82, Sn=118.71, Sb=121.76, Te=127.60,
                        I=126.90, Xe=131.29, Cs=132.91, Ba=137.33, La=138.91,
                        Ce=140.12, Pr=140.91, Nd=144.24, Pm=145.00, Sm=150.36,
                        Eu=151.96, Gd=157.25, Tb=158.93, Dy=162.50, Ho=164.93,
                        Er=167.26, Tm=168.93, Yb=173.04, Lu=174.97, Hf=178.49,
                        Ta=180.95, W=183.84, Re=186.21, Os=190.23, Ir=192.22,
                        Pt=195.08, Au=196.97, Hg=200.59, Tl=204.38, Pb=207.2,
                        Bi=208.98, Po=209.00, At=210.00, Rn=222.00, Fr=223.00,
                        Ra=226.00, Ac=227.00, Th=232.04, Pa=231.04, U=238.03,
                        Np=237.00, Pu=244.00, Am=243.00, Cm=247.00, Bk=247.00,
                        Cf=251.00, Es=252.00, Fm=257.00, Md=258.00, No=259.00,
                        Lr=262.00, Rf=261.00, Db=262.00, Sg=266.00, Bh=264.00,
                        Hs=269.00, Mt=268.00)

    mol = mol.copy()
    mol.filter(sel, _logger=False)
    mol.wrap("protein")
    mol.center()
    
    mass = np.array([_atomic_mass[i] for i in mol.element])
    t_mass = np.sum(mass)

    center_of_mass = np.sum(mol.coords * mass[:, None, None], axis=0) / t_mass
    dist_from_com = np.sqrt(np.sum((mol.coords - center_of_mass)**2, axis=1))
    rg = np.sqrt(np.sum(mass[: , None] * dist_from_com**2, axis=0) / t_mass)
    return rg[:, None]


if __name__ == '__main__':
    from htmd.molecule.molecule import Molecule
    # from htmd.projections.metric import Metric
    # met_rg = Metric(sim)
    mol = Molecule("./ref_files/MD_trajectory/structure.pdb")
    mol.read("./ref_files/MD_trajectory/structure.psf")
    mol.read("./ref_files/MD_trajectory/output.xtc")
    mol.wrap("protein")
    mol.align("protein")
    met = metricRG(mol)
