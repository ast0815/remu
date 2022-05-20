from remu import binning, plotting

with open("reco-binning.yml") as f:
    reco_binning = binning.yaml.full_load(f)

reco_binning.fill_from_csv_file("real_data.txt")

pltr = plotting.get_plotter(reco_binning)
pltr.plot_values()
pltr.savefig("real_data.png")

reco_binning.reset()
reco_binning.fill_from_csv_file("modelA_data.txt")

pltr = plotting.get_plotter(reco_binning)
pltr.plot_values()
pltr.savefig("modelA_data.png")

reco_binning.reset()
reco_binning.fill_from_csv_file("modelB_data.txt")

pltr = plotting.get_plotter(reco_binning)
pltr.plot_values()
pltr.savefig("modelB_data.png")

pltr = plotting.get_plotter(reco_binning)
reco_binning.reset()
reco_binning.fill_from_csv_file("real_data.txt")
pltr.plot_values(label="data", scatter=500)
reco_binning.reset()
reco_binning.fill_from_csv_file("modelA_data.txt")
pltr.plot_values(label="model A", scatter=500)
reco_binning.reset()
reco_binning.fill_from_csv_file("modelB_data.txt")
pltr.plot_values(label="model B", scatter=500)
pltr.legend()
pltr.savefig("compare_data.png")

with open("truth-binning.yml") as f:
    truth_binning = binning.yaml.full_load(f)

truth_binning.fill_from_csv_file("modelA_truth.txt")

pltr = plotting.get_plotter(truth_binning)
pltr.plot_values()
pltr.savefig("modelA_truth.png")

truth_binning.reset()
truth_binning.fill_from_csv_file("modelB_truth.txt")

pltr = plotting.get_plotter(truth_binning)
pltr.plot_values()
pltr.savefig("modelB_truth.png")

pltr = plotting.get_plotter(truth_binning)
truth_binning.reset()
truth_binning.fill_from_csv_file("modelA_truth.txt")
pltr.plot_values(label="model A", scatter=500)
truth_binning.reset()
truth_binning.fill_from_csv_file("modelB_truth.txt")
pltr.plot_values(label="model B", scatter=500)
pltr.legend()
pltr.savefig("compare_truth.png")
