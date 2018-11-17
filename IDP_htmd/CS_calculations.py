

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
	def sparta_launcher(pdbs, out="/tmp/"):
		import os
		all_files = " ".join(pdbs)
		os.makedirs(out, exist_ok=True)
		os.chdir(out)
		os.system("/shared/pablo/SPARTA+/sparta+ -in {} > {}/log.txt".format(all_files, out))

	@staticmethod
	def shiftx_launcher(pdbs, out="/tmp/", temperature=310):
		import os
		import multiprocessing as mp

		os.makedirs(out, exist_ok=True)
		command = "python2 /shared/pablo/shiftx2-linux/shiftx2.py -t {} -i ".format(temperature)
		pool = mp.Pool(processes=9) 
		all_data = [ [command + i, out+"/"+os.path.basename(i).split(".")[0]] for i in pdbs ]
		# for i in all_data:
		# 	_launch_worker(i)
		pool.map(_launch_worker, all_data)

	def compare_CS(self):
		pass

	@staticmethod
	def getNMRinfo(battery, nmr_type, seq, column=['shift'], atomType=['CB', 'CA']):
		dt = {
			'sparta': [np.dtype([('resid', '<i8'), ('resname', 'U1'), ('atomType', 'U3'), 
					('ss_shift', '<f8'), ('shift', '<f8'), ('rc_shift', '<f8'), 
					('hm_shift', '<f8'), ('ef_shift', '<f8'), ('sigma', '<f8')]), 27, None],
			'shiftx2': [np.dtype([('resid', '<i8'), ('resname', 'U1'), 
					('atomType', 'U4'), ('shift', '<f8')]), 1, ","]
			}

		gly_pos = np.where(np.array(list(seq)) == 'G' )[0]

		try:
			row_dt, header, delim = dt[nmr_type]
		except:
			return

		for idx, nmr_file in enumerate(battery):
			nmrData = np.genfromtxt(nmr_file, skip_header=header, delimiter=delim, dtype=row_dt)
			nmrData = np.asarray(nmrData.tolist(), dtype=row_dt)

			column = []
			for atom in atomType:
				data = CScalculations._get_cshift(nmrData, atom)
				if atom == 'CB':
					for i in gly_pos:
						data = np.insert(data, i, [0])
				column = np.hstack((column, data))
			if idx == 0: 
				all_data = np.vstack(column)
			else:
				all_data = np.column_stack((all_data, column))

		out = pd.DataFrame(all_data)
		atom = np.array([[j]*len(seq) for j in atomType]).flatten()
		out = out.set_index(atom, append=True)
		out.index.names = [ 'ids', 'atomName' ]
		out.index = out.index.droplevel(0)
		return out

	@staticmethod
	def _get_cshift(npdata, atomType='CA', column=['shift']):
	    return npdata[npdata['atomType'] == atomType]['shift']

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
