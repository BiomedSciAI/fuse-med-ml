import argparse
import os
from cvar_pyutils.ccc import submit_job

parser = argparse.ArgumentParser()
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--num_devices", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--bs", type=int, default=32)
args = parser.parse_args()
num_nodes = args.num_nodes
num_devices = args.num_devices
lr = args.lr
bs = args.bs

# Make Command
command_to_run = None
if (num_nodes > 1) or (num_devices > 1):
    os.environ["OMP_NUM_THREADS"] = "64"
    # command_to_run=f'pyutils-run /u/shatz/repos/fuse-med-ml/examples/fuse_examples/imaging/classification/mnist/ddp_example/simple_mnist_starter_ddp.py --num_nodes={num_nodes} --num_devices={num_devices} --lr={lr} --bs={bs}'
    command_to_run = f"pyutils-run /u/shatz/repos/fuse-med-ml/examples/fuse_examples/imaging/classification/knight/baseline/fuse_baseline.py --num_nodes={num_nodes} --num_devices={num_devices}"
else:
    # command_to_run=f'python /u/shatz/repos/fuse-med-ml/examples/fuse_examples/imaging/classification/mnist/ddp_example/simple_mnist_starter_ddp.py --num_nodes={num_nodes} --num_devices={num_devices} --lr={lr} --bs={bs}'
    command_to_run = "python /u/shatz/repos/fuse-med-ml/examples/fuse_examples/imaging/classification/knight/baseline/fuse_baseline.py"

submit_job(
    command_to_run=command_to_run,
    machine_type="x86",
    duration="1h",
    num_nodes=num_nodes,
    num_cores=16,
    num_gpus=num_devices,
    mem="120g",
    gpu_type="v100",
    # gpu_type='v100 || a100_80gb || a100_40gb',
    # mail_log_file_when_done='daniel.shats1@ibm.com',
    verbose_output=True,  # prints things like job id after submission to command line
)
