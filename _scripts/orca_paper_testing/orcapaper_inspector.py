from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

print_tensors_in_checkpoint_file(
    "/home/james/raynet/_models/Orca-paper/model.ckpt-1283529",
    tensor_name='',
    all_tensors=False,
    all_tensor_names=True
)