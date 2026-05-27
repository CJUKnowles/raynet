from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os
print_tensors_in_checkpoint_file(
    f"{os.getenv("RAYNET_PATH")}/_models/Orca-papermodel/model.ckpt-1283529",
    tensor_name='',
    all_tensors=False,
    all_tensor_names=True
)