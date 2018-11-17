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
