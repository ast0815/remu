from remu import binning, migration

builder = migration.ResponseMatrixArrayBuilder(1)

with open("../01/reco-binning.yml") as f:
    reco_binning = binning.yaml.full_load(f)
with open("../01/optimised-truth-binning.yml") as f:
    truth_binning = binning.yaml.full_load(f)

resp = migration.ResponseMatrix(reco_binning, truth_binning)

n_toys = 100
for i in range(n_toys):
    resp.reset()
    resp.fill_from_csv_file(
        ["modelA_data.txt", "modelB_data.txt"],
        weightfield="weight_%i" % (i,),
        rename={"reco_x_%i" % (i,): "reco_x"},
        buffer_csv_files=True,
    )
    resp.fill_up_truth_from_csv_file(
        ["../00/modelA_truth.txt", "../00/modelB_truth.txt"], buffer_csv_files=True
    )
    builder.add_matrix(resp)

builder.export("response_matrix.npz")
