from remu import binning
from remu import plotting

with open("reco-binning.yml", 'r') as f:
    reco_binning = binning.yaml.load(f)

reco_binning.fill_from_csv_file("real_data.txt")

plt = plotting.get_plotter(reco_binning)
plt.plot_values()
plt.savefig("real_data.png")

reco_binning.reset()
reco_binning.fill_from_csv_file("modelA_data.txt")

plt = plotting.get_plotter(reco_binning)
plt.plot_values()
plt.savefig("modelA_data.png")

reco_binning.reset()
reco_binning.fill_from_csv_file("modelB_data.txt")

plt = plotting.get_plotter(reco_binning)
plt.plot_values()
plt.savefig("modelB_data.png")

plt = plotting.get_plotter(reco_binning)
reco_binning.reset()
reco_binning.fill_from_csv_file("real_data.txt")
plt.plot_values(label="data", scatter=500)
reco_binning.reset()
reco_binning.fill_from_csv_file("modelA_data.txt")
plt.plot_values(label="model A", scatter=500)
reco_binning.reset()
reco_binning.fill_from_csv_file("modelB_data.txt")
plt.plot_values(label="model B", scatter=500)
plt.legend()
plt.savefig("compare_data.png")

with open("truth-binning.yml", 'r') as f:
    truth_binning = binning.yaml.load(f)

truth_binning.fill_from_csv_file("modelA_truth.txt")

plt = plotting.get_plotter(truth_binning)
plt.plot_values()
plt.savefig("modelA_truth.png")

truth_binning.reset()
truth_binning.fill_from_csv_file("modelB_truth.txt")

plt = plotting.get_plotter(truth_binning)
plt.plot_values()
plt.savefig("modelB_truth.png")

plt = plotting.get_plotter(truth_binning)
truth_binning.reset()
truth_binning.fill_from_csv_file("modelA_truth.txt")
plt.plot_values(label="model A", scatter=500)
truth_binning.reset()
truth_binning.fill_from_csv_file("modelB_truth.txt")
plt.plot_values(label="model B", scatter=500)
plt.legend()
plt.savefig("compare_truth.png")
