
import argparse
import logging
import os
import numpy as np
from natsort import natsorted
from htmd.adaptive.adaptive import epochSimIndexes

class AdaptiveAnalysis:
    def __init__(self, input_folder, output_folder, analysis_function, sims=None,
        precalc_metric=None, epoch_analysis=None, sim_analysis=None, low_memory_usage=False, test=False):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.analyze_function = analysis_function
        self.precalc_metric = precalc_metric
        self.current_epoch = None
        self.current_analysis = None
        self.precalc_data = None
        self.precalculated_data = False
        self.analysis_type = "epoch"
        self.epoch_analysis = epoch_analysis
        self.sim_analysis = sim_analysis
        self.epoch_sim_indexes = None
        self.sims_number = 0
        self.associated_metrics = {}
        self.associated_data = {}
        self.tmp_associated_data = {}
        self.low_memory_usage = low_memory_usage
        if isinstance(sims, np.ndarray):
            self._sims = sims
        else:
            self._sims = self._getsimlist(input_folder)
        self.epoch_sim_indexes = epochSimIndexes(self._sims)
        #self._detect_epochs() ## Sets both self.current epoch and self.current_analysis
        self.test = test

    def _createLogger(self, epoch):
        logger = logging.getLogger(__name__)
        logger.setLevel("WARNING")
        filename = f'{self.output_folder}/{self.analysis_type[0]}{epoch}_log.txt'
        with open(filename, "w") as myfile:
            myfile.write("")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        fh = logging.FileHandler(filename)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _detect_epochs(self, prefix):
        import re
        regex = f"^{prefix}([0-9]+)*"
        regex2 = f"^e([0-9]+)*"

        def detect_current_epoch(data_list, type_info="model"):

            if len(data_list) > 1:
                data_epoch = os.path.basename(data_list[-1])
                return int(re.findall(regex2, data_epoch)[0])
            else:
                return 1

        from glob import glob

        epoch_list = glob(f"{self.input_folder}/data/*")
        epoch_list = natsorted(epoch_list)
        analysis_list = glob(f"{self.output_folder}/{self.analysis_type[0]}*log.txt") ## Marks if the analysis have been already performed
        analysis_list = natsorted(analysis_list)

        self.current_analysis = [int(re.findall(regex, os.path.basename(i))[0]) for i in analysis_list]
        self.current_epoch = detect_current_epoch(epoch_list, "sims")

    def perform_analysis(self, fstep=None, basedata=None, analysis_step=5, skip=1, clusters=0, ticadim=0, ticalag=20,
        macronum=2, modellag=5, modelunits="frames", data2combine=None, **kwargs):

        if self.analysis_type == "epoch":
                self._detect_epochs("e")
                start = 1
        elif self.analysis_type == "sims":
                self._detect_epochs("s")
                start = analysis_step

        if not self.epoch_analysis and self.analysis_type[0] == "s":
            self.epoch_analysis = np.arange(start, len(self._sims), analysis_step)[::-1]
        elif not self.epoch_analysis and self.analysis_type[0] == "e":
            self.epoch_analysis = np.arange(start, np.max(list(self.epoch_sim_indexes.keys())) + 1, analysis_step)
        else:
            self.epoch_analysis = sorted(self.epoch_analysis, reverse=True)

        print(self.epoch_analysis)
        for i in self.epoch_analysis[::-1]:
            if i not in self.current_analysis:
                print(f"Analyzing {self.analysis_type} - {i}")
                model = self._createMSM(i, self.output_folder, basedata=basedata, skip=skip, clusters=clusters, ticadim=ticadim, ticalag=ticalag,
                    macronum=macronum, modellag=modellag, modelunits=modelunits, fstep=fstep, data2combine=data2combine)
                #Include self.tmp_associated_data,
                self.analyze_function(i, model, self.output_folder, self.tmp_associated_data, **kwargs)
                self.logger = self._createLogger(i)
                self.current_analysis.append(i)

    def _getsimlist(self, folder):
        from htmd.simlist import simlist
        from glob import glob
        simfolders = glob(f'{folder}/filtered/*/')

        tmp_sims = []
        #To avoid problems while merging multiples data sources
        clean_names = set([i.split("/")[-2] for i in simfolders ])
        for sim in simfolders:
            tmp_name = sim.split("/")[-2]
            if tmp_name in clean_names:
                tmp_sims.append(sim)
                clean_names.remove(tmp_name)
        simfolders = tmp_sims

        all_folders = glob(folder)[0]
        sims = simlist(simfolders, f'{all_folders}/filtered/filtered.pdb')
        return sims

    def _precalculateData(self, metricData, folder, skip=1, fstep=None):
        from htmd.projections.metric import Metric

        max_epoch = max(self.epoch_analysis)
        max_epoch_sim = np.concatenate(np.array([self.epoch_sim_indexes[i] for i in range(1, max_epoch + 1) if i in list(self.epoch_sim_indexes.keys())]))

        if self.test:
            sims = self._sims[0:100]
        else:
            sims = np.array([self._sims[i] for i in max_epoch_sim])

        metr = Metric(sims, skip=skip)
        metr.set(metricData)
        data = metr.project()
        if fstep:
            data.fstep = fstep
        return data

    def _createMSM(self, epoch, output_folder, basedata=None, skip=1, clusters=0, ticadim=0, ticalag=20, macronum=2, modellag=5, modelunits="frames", fstep=None, data2combine=None):
        from htmd.projections.tica import TICA
        from sklearn.cluster import MiniBatchKMeans
        from htmd.model import Model

        try:
            model = Model(file=f"{output_folder}/{self.analysis_type[0]}{epoch}_model.dat")

            if (model.macronum != macronum or model.lag != modellag):
                model.markovModel(modellag, macronum, units=modelunits)
            print("Model loaded")
        except:
            if not self.precalculated_data and not self.low_memory_usage:
                print("Calculating PRECALC DATA")
                precalc_data = self._precalculateData(self.precalc_metric, self.input_folder, fstep=fstep, skip=skip)
                self.precalc_data = precalc_data
                self.precalculated_data = True

            if self.analysis_type == "epoch" and not self.low_memory_usage:
                epoch_sim = np.concatenate(np.array([self.epoch_sim_indexes[i] for i in range(1, epoch + 1) if i in list(self.epoch_sim_indexes.keys())]))
                drop_traj_idx = np.ones(self.precalc_data.numTrajectories)
                drop_traj_idx[epoch_sim] = 0
                drop_idx = np.where(drop_traj_idx == 1)[0]
            elif self.analysis_type == "sims" and not self.low_memory_usage:
                drop_traj_idx = np.ones(self.precalc_data.numTrajectories)
                no_drop_idx = np.arange(1, epoch)
                drop_traj_idx[no_drop_idx] = 0
                drop_idx = np.where(drop_traj_idx == 1)[0]

            if not self.low_memory_usage:
                data = self.precalc_data.copy()
                data.dropTraj(idx=drop_idx)
                data.dropTraj()

            if basedata:
                from htmd.projections.metric import MetricData
                r_fit = self._fitBaseline(data, basedata)
                data = MetricData(dat=r_fit, simlist=data.simlist)
            elif ticadim and not self.low_memory_usage:
                tica = TICA(data, ticalag)
                data = tica.project(ticadim)
            elif ticadim and self.low_memory_usage:
                from htmd.projections.metric import Metric
                if self.analysis_type == "epoch":
                    epoch_sim = np.concatenate(np.array([self.epoch_sim_indexes[i] for i in range(1, epoch + 1) if i in list(self.epoch_sim_indexes.keys())]))
                else:
                    epoch_sim = range(0, epoch)
                metr = Metric(self._sims[epoch_sim], skip=skip)
                metr.set(self.precalc_metric)
                tica = TICA(metr, ticalag)
                data = tica.project(ticadim)
            if not clusters:
                clusters = self._numClusters(data.numFrames)

            if data2combine:
                try:
                    print("Adding extra dimension")
                    data2combine_copy = data2combine.copy()
                    data2combine_copy.dropTraj(keepsims=data.simlist)
                    data.combine(data2combine_copy)
                except Exception as e:
                    print("Could not combined data", str(e))

            data.cluster(MiniBatchKMeans(clusters), mergesmall=5)
            model = Model(data)
            model.markovModel(modellag, macronum, units=modelunits)
            model.save(f"{output_folder}/{self.analysis_type[0]}{epoch}_model.dat")

        for name, met in self.associated_metrics.items():
            try:
                self.associated_data[name]
            except:
                print(f"Calcualtion associted data - {name.upper()}")
                assoc_data = self._precalculateData(met, self.input_folder, fstep=fstep, skip=skip)
                self.associated_data[name] = assoc_data

        for name, data in self.associated_data.items():
            tmp_data = data.copy()
            tmp_data.dropTraj(keepsims=model.data.simlist)
            self.tmp_associated_data[name] = tmp_data

        return model

    def _numClusters(self, numFrames):
        """ Heuristic that calculates number of clusters from number of frames """
        import numpy as np
        K = int(max(np.round(0.6 * np.log10(numFrames / 1000) * 1000 + 50), 100))  # heuristic
        if K > numFrames / 3:  # Ugly patch for low-data regimes ...
            K = int(numFrames / 3)
        return K

    def _fitBaseline(self, data, basedata, ticalag=25, ticadim=4, ticaunits='ns', tica=False):
        from htmd.projections.tica import TICA

        basetica = TICA(basedata, ticalag, units=ticaunits)
        basetica.tic.set_params(dim = ticadim)
        tmp_data = np.concatenate(data.dat)

        r_fit = basetica.tic.transform(tmp_data)
        r_fit = data.deconcatenate(r_fit)
        return r_fit

    ## Make setter for epoch_analysis



def corruptMetric(mol):
    from moleculekit.projections.metriccoordinate import MetricCoordinate
    coor_dat = MetricCoordinate("backbone and name CA").project(mol)

    return np.mean(np.abs(coor_dat), axis=1) > 1000

def removeCorrupted():
    from htmd.simlist import simlist
    from htmd.projections.metric import Metric
    from os import path
    from glob import glob
    import shutil

    print("Removing Corrupted Simulations")
    try:
        sims = simlist(glob("./filtered/*/"), "./filtered/filtered.pdb")
    except:
        return
    met = Metric(sims)
    met.set(corruptMetric)
    dat = met.project()
    for i, s in zip(dat.dat, dat.simlist):
        if np.sum(i):
            pt = path.dirname(s.trajectory[0])
            shutil.move(pt, f"/tmp/{pt}")
