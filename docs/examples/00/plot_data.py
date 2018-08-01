from remu import binning

with open("reco-binning.yml", 'r') as f:
    reco_binning = binning.yaml.load(f)

reco_binning.fill_from_csv_file("real_data.txt")
reco_binning.plot_values("real_data.png", variables=(None, None))
reco_binning.plot_values("real_data_abs.png", variables=(None, None), divide=False)

reco_binning.reset()
reco_binning.fill_from_csv_file("modelA_data.txt")
reco_binning.plot_values("modelA_data.png", variables=(None, None))
reco_binning.plot_values("modelA_data_abs.png", variables=(None, None), divide=False)

reco_binning.reset()
reco_binning.fill_from_csv_file("modelB_data.txt")
reco_binning.plot_values("modelB_data.png", variables=(None, None))
reco_binning.plot_values("modelB_data_abs.png", variables=(None, None), divide=False)

with open("truth-binning.yml", 'r') as f:
    truth_binning = binning.yaml.load(f)

truth_binning.fill_from_csv_file("modelA_truth.txt")
truth_binning.plot_values("modelA_truth.png", variables=(None, None))
truth_binning.plot_values("modelA_truth_abs.png", variables=(None, None), divide=False)

truth_binning.reset()
truth_binning.fill_from_csv_file("modelB_truth.txt")
truth_binning.plot_values("modelB_truth.png", variables=(None, None))
truth_binning.plot_values("modelB_truth_abs.png", variables=(None, None), divide=False)
