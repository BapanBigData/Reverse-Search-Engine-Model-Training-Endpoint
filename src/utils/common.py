import torch
import numpy as np
import time


def set_seed(seed_value: int = 47) -> None:
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def get_unique_filename(filename, ext):
    return time.strftime(f"{filename}_%Y_%m_%d_%H_%M.{ext}")


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    
    print(f"Train time on {device}: {total_time:.3f} seconds")
    
    return total_time

