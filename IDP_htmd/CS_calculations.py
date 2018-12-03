

class CScalculations():
	def __init__(self, pdb_files, temparature=310, method="sparta"):
		self.orig_files = pdb_files
		self.method = method

	def lauch_works(self):
		lauchers = {"sparta": self.sparta_launcher,
								"shiftx": self.shiftx_launcher
								}
		lauchers[self.method](self.orig_files)

	@staticmethod
	def sparta_launcher(folder, out="/tmp/"):
		import os
		# all_files = " ".join(pdbs)
		os.makedirs(out, exist_ok=True)
		os.chdir(out)
		os.system("/shared/pablo/SPARTA+/sparta+ -in {}/*pdb > {}/log.txt".format(folder,
																																		 out))

	@staticmethod
	def shiftx_launcher(pdbs, out="/tmp/", temperature=310):
		import os
		import multiprocessing as mp

		os.makedirs(out, exist_ok=True)
		command = "python2 /shared/pablo/shiftx2-linux/shiftx2.py -t {} -u -i".format(
				temperature)
		pool = mp.Pool(processes=19)
		all_data = [[command + i, out+"/" +
								 os.path.basename(i).split(".")[0]] for i in pdbs]
		pool.map(_launch_worker, all_data)

	@staticmethod
	def getNMRinfo(battery, nmr_type, seq,
								 atomType=['CB', 'CA'], sparta_fix=[0]):
		import numpy as np
		import pandas as pd
		dt = {
				'sparta': [np.dtype([('resid', '<i8'), ('resname', 'U1'),
														 ('atomType', 'U3'), ('ss_shift',
																									'<f8'), ('shift', '<f8'),
														 ('rc_shift', '<f8'), ('hm_shift',
																									 '<f8'), ('ef_shift', '<f8'),
														 ('sigma', '<f8')]), 27, None],
				'shiftx2': [np.dtype([('chain', 'U1'),('resid', '<i8'), ('resname', 'U1'),
															('atomType', 'U4'), ('shift', '<f8')]), 1, ","]
		}

		if nmr_type not in ['sparta', 'shiftx2']:
			print("Suported types for 'sparta' or 'shiftx2'")
			return

		gly_pos = np.where(np.array(list(seq)) == 'G')[0]
		pro_pos = np.where(np.array(list(seq)) == 'P')[0]

		try:
			row_dt, header, delim = dt[nmr_type]
		except:
			return

		for idx, nmr_file in enumerate(battery):
			nmrData = np.genfromtxt(nmr_file, skip_header=header,
															delimiter=delim, dtype=row_dt)
			nmrData = np.asarray(nmrData.tolist(), dtype=row_dt)

			column = []
			for atom in atomType:
				if atom == 'H' and nmr_type == 'sparta':
					atom = 'HA'
				data = CScalculations._get_cshift(nmrData, atom)
				if atom in ['CB', 'HA']:
					for i in gly_pos:
						data = np.insert(data, i, [0])
				if atom == 'N' and nmr_type=='sparta':
					for fix in sparta_fix:
						data = np.insert(data, fix, [0])
				if atom in ['N', 'H']:
					for i in pro_pos:
						data = np.insert(data, i, [0])
				column = np.hstack((column, data))
			if idx == 0:
				all_data = np.vstack(column)
			else:
				all_data = np.column_stack((all_data, column))

		out = pd.DataFrame(all_data)
		atom = np.array([[j]*len(seq) for j in atomType]).flatten()
		out = out.set_index(atom, append=True)
		out.index.names = ['ids', 'atomName']
		out.index = out.index.droplevel(0)
		return out

	@staticmethod
	def _get_cshift(npdata, atomType='CA'):
		return npdata[npdata['atomType'] == atomType]['shift']

	@staticmethod
	def plotCSdiff(ax, data, labels, limit, title=None, xlabels=None):
		style_pallete = [{'edgecolor':'#3b5998', 'color':'#3b5998'},
			{'edgecolor':'red', 'color':(0, 0, 0, 0)}]

		import matplotlib.pyplot as plt
		for idx, (lab, dat) in enumerate(zip(labels, data)):
				ax.bar(range(len(dat)), dat, edgecolor=style_pallete[idx]['edgecolor'],
							 color=style_pallete[idx]['color'], label=lab)

		ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
		ax.set_ylabel("Chemichal shift difference(p.p.m)")
		ax.set_xlabel("Residue Index")

		if not xlabels:
			ax.set_xticks(range(0, len(data[0])+1, 5))
			ax.set_xticklabels(range(0, len(data[0])+1, 5))
		else:
			ax.set_xticklabels(xlabels, rotation=45, ha="center")
			ax.set_xticks(range(0, len(xlabels)))
		if limit:
			_ = [ax.hlines(i, -1, len(data[0])+1, colors="red",
										 linestyles="dashed", linewidth=1) for i in limit]

		if title:
			ax.set_title(title)
		return ax

	@staticmethod
	def plotCSdata(data, atomtypes, labels=None, title=None, plot=True, save=None, **kwargs):
		import matplotlib.pyplot as plt
		import numpy as np

		limits = {'CA': [0.4, -0.4], 'CB': [0.5, -0.5], 
		'N': [1.1, -1.1], 'H': [0.1, -0.1], }
		f, axes = plt.subplots(ncols=1, nrows=len(
			atomtypes), figsize=(13, 5*(len(atomtypes))))
		if title:
			f.suptitle(title)

		# import pdb; pdb.set_trace();
		for atomtype, ax in zip(atomtypes, axes):
			chunck = [dat.loc[atomtype] for dat in data]
			CScalculations.plotCSdiff(ax, chunck, labels,
																limit=limits[atomtype], title=atomtype, **kwargs)
		
		if plot:
			plt.show()
		
		if save:
			plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.2)

	@staticmethod
	def cleanBMRB(filename, outdir, correct_resid=0, rmin=None, rmax=None):
		import pandas as pd
		with open(filename, 'r') as myfile:
			headers = []
			values = []
			for line in myfile.readlines():
				line = line.strip()
				line = line.replace(". ", '')
				line = ' '.join(line.split())
				line = line.split(" ")
				values.append(tuple(line)) if len(line)>0 else None

		output = pd.DataFrame(values)
		output.drop(columns=[0, 1, 2, 3, 7, 8, 10, 11, 12, 13], index=1, inplace=True)
		output.columns = ['resid', 'resname', 'atom','cs_shift']
		output.to_csv(outdir, index=False)
		myfile.close()

		output = pd.read_csv(outdir)
		output.resid = output.resid.apply(lambda x: x+correct_resid)
		if rmin:
			output = output[output.resid >= rmin]
		if rmax:
			output = output[output.resid <= rmax]
		output.sort_values(by=['atom', 'resid'], inplace=True)
		output.to_csv(outdir, index=False)
		return output

def _launch_worker(info):
		import os
		command = "{} -o {}.cs >> /tmp/log.txt".format(info[0], info[1])
		os.system(command)


if __name__ == "__main__":
		from glob import glob
		import numpy as np
		import pandas as pd

		test_files = glob("/shared/pablo/test_pdbs/*pdb")
		sparta = CScalculations(test_files, "sparta")
		# sparta.lauch_works()

		shiftx = CScalculations(test_files, "shiftx")
		# shiftx.lauch_works()

		# CScalculations.sparta_launcher(test_files, "/shared/pablo/cs_output",)
		# CScalculations.shiftx_launcher(test_files, "/shared/pablo/cs_sh_output", 310)

		sparta_files = glob("/shared/pablo/cs_output/*pred.tab")
		shiftx_files = glob("/shared/pablo/cs_sh_output/*cs")
		from htmd.ui import Molecule
		seq = Molecule(test_files[0]).sequence()['X']
		sp_info = CScalculations.getNMRinfo(sparta_files, 'sparta', seq)
		sh_info = CScalculations.getNMRinfo(shiftx_files, 'shiftx2', seq)
