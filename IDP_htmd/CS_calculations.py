import matplotlib.pyplot as plt


class CScalculations():
    """Provides methods for handling chemical shift calculations
    and associated plottfin functions
    """
    
    def __init__(self):
        pass
        # self.orig_files = pdb_files
        # self.method = method

    @staticmethod
    def sparta_launcher(path, out_folder="./SPARTA_data", out="/tmp/"):
        """Launch SPARTA+
        Perfoms Chemical shift prediction for all the pdb files found in the folder using SPARTA+
        
        Parameters
        ----------
        pdbs : str
            Array with the pdb files paths to be predicted
        out : str, optional
            Output directory where predictions will be stored
        """
        import os
        import shutil
        from glob import glob
        from time import sleep
        os.makedirs(out, exist_ok=True)
        pwd = os.getcwd()
        os.chdir("/tmp")

        os.system(f"/shared/pablo/SPARTA+/sparta+ -in {path}/*pdb -out {out_folder}/pred.tab -outS {out_folder}/struct.tab> {out}/log.txt")
        keep_scanning, fail_attempts = True, 0
        os.makedirs(out_folder, exist_ok=True)
        sleep(5)
        while keep_scanning:
            outfiles = glob(f"{path}/*pred.tab")
            out_struct = glob(f"{path}/*struct.tab")
            print(len(outfiles))
            if len(outfiles) == 0: fail_attempts += 1
            if fail_attempts > 5: keep_scanning = False
            _ = [shutil.move(i, out_folder) for i in outfiles]
            _ = [shutil.move(i, out) for i in out_struct]

        os.chdir(pwd)

    @staticmethod
    def shiftx_launcher(pdbs, out="/tmp/", temperature=310):
        """Launch SHIFTX2
        Perfoms chemical shift predictions for all pdb file found in the folder using the SHIFTX2 software
        
        Parameters
        ----------
        pdbs : str[]
            Array with the pdb files paths to be predicted
        out : str, optional
            Output directory where predictions will be storednal
            Description
        temperature : int, optional
            Temperature in K of the simulations
        """
        import os
        import multiprocessing as mp

        os.makedirs(out, exist_ok=True)
        #Modified version of the sotware to be parrallelizable
        command = f"python2 /shared/pablo/software/shiftx2-linux/shiftx2.py -t {temperature} -i"
        pool = mp.Pool(processes=10)
        all_data = [[f"{command}{i}", f"{out}/{os.path.basename(i).split('.')[0]}"] for i in pdbs]
        pool.map(_launch_worker, all_data)

    @staticmethod
    def getNMRinfo2(battery, atomType=['CB', 'CA'], sparta_fix=[0]):
        import pandas as pd
        import numpy as np
        for idx, i in enumerate(battery):
            tmp = pd.read_csv(i)
            if idx == 0:
                out = tmp[np.isin(tmp.ATOMNAME, atomType)]
                out.rename(str, columns={out.columns[-1]: idx}, inplace=True)

            else:
                tmp_data = np.array(tmp[tmp.columns[-1]][np.isin(tmp.ATOMNAME, atomType)])
                #print(len(tmp_data))
                out[idx] = tmp_data
        out.set_index(['CHAIN', 'NUM', 'RES', 'ATOMNAME'], inplace=True)
        return out

    @staticmethod
    def getNMRinfo(battery, nmr_type, seq, start_idx=0,
                    atomType=['CB', 'CA'], sparta_fix=[0]):
        """Extracts CS from files
        Reads output of either sparta or shiftx
        
        Parameters
        ----------
        battery : str[]
            Array of fils outputed by either sparta or shiftx2
        nmr_type : str
            NMR prediction type. It can be either "sparta" or "shiftx"
        seq : str
            Aminoacid sequence of the predicted aminoacids
        atomType : list, optional
            Atomtypes to be read
        sparta_fix : list, optional
            Description
        
        Returns
        -------
        pandas.DataFrame
            Dataframe nxm dimenions, where m is the number of files and n the length of atomtypes x length the sequence
        """
        import numpy as np
        import pandas as pd
        dt = {'sparta': [np.dtype([('resid', '<i8'), 
                                    ('resname', 'U1'),
                                    ('atomType', 'U3'), 
                                    ('ss_shift','<f8'),
                                    ('shift', '<f8'),
                                    ('rc_shift', '<f8'), 
                                    ('hm_shift','<f8'), 
                                    ('ef_shift', '<f8'),
                                    ('sigma', '<f8')]), 27, None],
                'shiftx2': [np.dtype([('resid', '<i8'),
                                        ('chain', 'U1'), 
                                        ('resname', 'U1'),
                                        ('atomType', 'U4'), 
                                        ('shift', '<f8')]), 1, ","]
        }

        if nmr_type not in ['sparta', 'shiftx2']:
            raise TypeError("Suported types are 'sparta' or 'shiftx2'")

        # Some aminoacids do not have some of the common backbone atoms
        # For those cases zero is returned
        gly_pos = np.where(np.array(list(seq)) == 'G')[0]
        pro_pos = np.where(np.array(list(seq)) == 'P')[0]
        resid_idx = []
        atom_idx = []

        try:
            row_dt, header, delim = dt[nmr_type]
        except:
            return

        for idx, nmr_file in enumerate(battery):
            nmrData = np.genfromtxt(nmr_file, skip_header=header, delimiter=delim, dtype=row_dt)
            nmrData = np.asarray(nmrData.tolist(), dtype=row_dt)

            column = []
            for atom in atomType:
                if atom == 'H' and nmr_type == 'sparta':
                    atom = 'HA'
                data = CScalculations._get_cshift(nmrData, atom)
                if atom in ['CB', 'HA']:
                    for i in gly_pos:
                        data = np.insert(data, i, [0])
                # if atom == 'N' and nmr_type=='sparta':
                #     for fix in sparta_fix:
                #         data = np.insert(data, fix, [0])
                # if atom in ['N', 'H']:
                #     for i in pro_pos:
                #         data = np.insert(data, i, [0])
                column = np.hstack((column, data))
                if idx == 0:
                    atom_idx.extend([atom]*len(data))
                    resid_idx.extend(nmrData[nmrData['atomType'] == atom]['resid'].tolist())
            if idx == 0:
                all_data = np.vstack(column)
            else:
                all_data = np.column_stack((all_data, column))

        out = pd.DataFrame(all_data)
        # atom = np.array([[j]*len(seq) for j in atomType]).flatten()
        atom_idx = np.array(atom_idx).ravel()
        out = out.set_index(np.array(resid_idx) + start_idx, append=True)
        out = out.set_index(atom_idx, append=True)
        out.index.names = ['ids', 'resid', 'atomName']
        out.index = out.index.droplevel(0)
        return out

    @staticmethod
    def _get_cshift(npdata, atomType='CA'):
        return npdata[npdata['atomType'] == atomType]['shift']

    @staticmethod
    def plotCSdiff(ax, data, std_data, label, limit, style=0, title=None, xlabels=None, legend=True):
        """Summary
        
        Parameters
        ----------
        ax : pyplot.ax
            Ax to plot the data
        data : TYPE
            Array with the difference
        labels : TYPE
            Description
        limit : TYPE
            Description
        title : None, optional
            Description
        xlabels : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        style_pallete = [{'edgecolor':'#3b5998', 'color':'#3b5998'},
            {'edgecolor':'red', 'color':(0, 0, 0, 0)}]

        ax.bar(range(len(data)), data, edgecolor=style_pallete[style]['edgecolor'],
                        color=style_pallete[style]['color'], label=label)
        # ax.errorbar(range(len(data)), data, std_data,lw=1, c='k', linestyle="", capsize=3)

        if legend: ax.legend(loc=0, shadow=True, ncol=2)
        if title == "CA": title = r"$C\alpha$"
        if title == "CB": title = r"$C\beta$"
        ax.set_ylabel(f"{title}\n Chemichal shift difference(p.p.m)")
        # ax.set_xlabel("Residue")

        if not xlabels:
            ax.set_xticks(range(0, len(data)+1, 5))
            ax.set_xticklabels(range(0, len(data)+1, 5))
        else:
            ax.set_xticklabels(xlabels, rotation=45, ha="center")
            ax.set_xticks(range(0, len(xlabels)))
        if limit:
            _ = [ax.hlines(i, -1, len(data)+1, colors="red",
                                         linestyles="dashed", linewidth=1) for i in limit]
        return ax

    @staticmethod
    def plotCSdata(mean_data, std_data, atomtypes, label=None, save=None, axes=None,**kwargs):
        """Plot CS differences 
        
        Parameters
        ----------
        data : []
            Array containing a list for each atomtype differences 
        atomtypes : []
            Array of atomtype where to be compared. It must have the same length of data
        labels : None, optional
            Description
        title : None, optional
            Description
        plot : bool, optional
            Description
        save : None, optional
            Description
        **kwargs
            Description
        """
        import matplotlib.pyplot as plt
        import numpy as np

        limits = {'CA': [0.4, -0.4], 'CB': [0.5, -0.5], 
        'N': [1.1, -1.1], 'H': [0.1, -0.1], }
        if axes is None:
            f, axes = plt.subplots(ncols=1, nrows=len(
                atomtypes), figsize=(13, 5*(len(atomtypes))))
        
        if len(atomtypes) == 1: axes = [axes]
        
        if label is None:
            label = [f"AtomType {i}" for i in range(len(mean_data))]

        new_axes = []
        for atomtype, m_data, s_data, ax in zip(atomtypes, mean_data, std_data, axes):
            # chunck = [dat.loc[atomtype] for dat in data]
            n_ax = CScalculations.plotCSdiff(ax, m_data, s_data, label,
                limit=limits[atomtype], title=atomtype, **kwargs)
            new_axes.append(n_ax)
        
        plt.subplots_adjust(hspace=.35)
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)
        return new_axes 

    @staticmethod
    def cleanBMRB(filename, outdir=None, correct_resid=0, rmin=None, rmax=None):
        """Read BMRB file
        The file has to be previously trimmed by other means to only have 
        the chemical shift section
        Parameters
        ----------
        filename : str
            BMRB text files
        outdir : str
            Folder to write the ooutput
        correct_resid : int, optional
            Correction of the index of the aminoacids
        rmin : int, optional
            Starting residue 
        rmax : int, optional
            Define the last residue number of the strecht
        Returns
        -------
        pandas.Dataframe
            Contains columns for CS, atomtype, atomname, residue number and residue name
        """
        import pandas as pd
        import numpy as np
        with open(filename, 'r') as myfile:
            values = []
            for line in myfile.readlines():
                line = line.strip()
                line = line.replace(". ", '')
                line = ' '.join(line.split())
                line = line.split(" ")
                values.append(tuple(line)) if (len(line)>=10 and str.isdigit(line[2])) else None
        myfile.close()

        output = pd.DataFrame(values)
        columns = [0, 2, 3, 7, 8, 10, 11, 12] 
        
        if output.shape[1] > 14:
            columns += list(range(13, output.shape[1]))

        output.drop(columns=columns, index=1, inplace=True)
        output.columns = ['chain', 'resid', 'resname', 'atom','cs_shift']

        output.to_csv("/tmp/test.pd", index=False)
        output = pd.read_csv("/tmp/test.pd")

        output.resid = output.resid.apply(lambda x: x+correct_resid)
        if rmin:
            output = output[output.resid >= rmin]
        if rmax:
            output = output[output.resid <= rmax]
        output.sort_values(by=['chain', 'atom', 'resid'], inplace=True)
        if outdir:
            output.to_csv(outdir, index=False)
        return output

def _launch_worker(info):
    #To allow paralelization
    import os
    command = "{} -o {}.cs >> /tmp/log.txt".format(info[0], info[1])
    os.system(command)


if __name__ == "__main__":
    from glob import glob
    import numpy as np
    import pandas as pd

    test_files = glob("/shared/pablo/test_pdbs/*pdb")

    sparta_files = glob("/shared/pablo/cs_output/*pred.tab")
    shiftx_files = glob("/shared/pablo/cs_sh_output/*cs")
    from htmd.ui import Molecule
    seq = Molecule(test_files[0]).sequence()['X']
    sp_info = CScalculations.getNMRinfo(sparta_files, 'sparta', seq)
    sh_info = CScalculations.getNMRinfo(shiftx_files, 'shiftx2', seq)
