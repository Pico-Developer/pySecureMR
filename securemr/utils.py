import os
import subprocess as commands


__all__ = ["run", "DEBUG_QNN", "TORCH_INSTALLED"]


def run(cmd) -> None:
    """Run bash command with print to stdout.
    
    Args:
        cmd_list: ["bash",  tmp_shell]
    """
    print(f"\033[0;33m>> {cmd}\033[0m")
    (status, output) = commands.getstatusoutput(cmd)
    print(output)
    if status != 0:
        raise RuntimeError(f"Faild to execute {cmd}")


DEBUG_QNN = bool(os.getenv("DEBUG_QNN", "0") == "1")

try:
    import torch
    import torchvision
    TORCH_INSTALLED = True
except ImportError as e:
    TORCH_INSTALLED = False
