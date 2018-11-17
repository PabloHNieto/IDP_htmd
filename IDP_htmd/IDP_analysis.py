
def analyze_folder(folder=None, out_folder="/tmp", metrics=None, fes=True, sk=1, clu=500, ticadim=5,
	model_lag=10, macro_N=10, save=False):
	from htmd.ui import simlist, Metric, TICA, Model, MiniBatchKMeans
	from glob import glob
	import os
	
	try:
		os.mkdir(out_folder)
	except:
		print("Folder already exists")

	print(metrics)
	sims = glob(folder + 'filtered/*/')
	fsims = simlist(sims, folder+'filtered/filtered.pdb')
	metr = Metric(fsims, skip=1)
	metr.set(metrics)
	data = metr.project()

	if ticadim:
		tica = TICA(data, 20)
		out_data = tica.project(ticadim)
	else:
		out_data = data

	x = True
	while x:
		try:
			out_data.cluster(MiniBatchKMeans(n_clusters=clu), mergesmall=5)
			x = False
		except:
			print("Trying again")

	model = Model(out_data)
	model.plotTimescales(plot=False, save=out_folder+"1_its.png")
	model.markovModel(model_lag, macro_N, units='ns')
	if fes:
		model.plotFES(0, 1, temperature=310, states=True,
			plot=False, save=out_folder+"2_fes.png")

	if save:
		model.save(out_folder + "model.dat")

	return model


def aux_plot(model, metric, mol, plot_func, method, **kwargs):
	from IDP_htmd.IDP_model import get_data
	from htmd.ui import getStateStatistic
	data = get_data(model, metric)
	data_summary = getStateStatistic(model, data, 
		method=method, states=range(model.macronum),
		statetype="macro")
	plot_func(data_summary, mol, **kwargs)


def bootstrap_model (data, rounds, folder, model_function=None, fraction=0.8, 
    clusters=500, lag=15, macroN=5):
  import os
  from htmd.ui import MiniBatchKMeans, Model
  print("Folder")
  try:
    os.mkdir(folder)
  except:
    print(folder + " already created")
  for boot_round in range(rounds):
    round_dir = "{}{}_round/".format(folder, boot_round + 1)
    print(round_dir)
    try:
        os.mkdir(round_dir)
    except:
        print(round_dir + " already created")

    dataBoot = data.bootstrap(fraction)
    x = True
    dataBoot.cluster(MiniBatchKMeans(n_clusters=clusters), mergesmall=5)
    while (x):
      try:
        dataBoot.cluster(MiniBatchKMeans(n_clusters=clusters), mergesmall=5)
        x = False
      except:
        print("Trying again")

    # Model generation
    model = Model(dataBoot)
    model.plotTimescales(plot=False, save=round_dir+"1_its.png")
    model.markovModel(lag, macroN, units='ns')
    model.plotFES(0, 1, temperature=310, states=True, 
      plot=False, save=round_dir+"2_fes.png")
    model.save(round_dir + "model.dat")

    # Model Postanalysis
    if model_function:
      model_function(model, round_dir)

