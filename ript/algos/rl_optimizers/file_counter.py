import os
import fcntl
import uuid
import torch.distributed as dist

class FileGlobalCounter:
    """
    A counter stored in a file for distributed coordination among processes.
    This is used for early stopping in rollout generation.
    """
    def __init__(self, filename):
        self.filename = filename
        # Create the file if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                f.write("0")
    
    def reset(self, value=0):
        """Reset the counter to the given value"""
        with open(self.filename, "w") as f:
            f.write(str(value))
    
    def update(self, increment=1):
        """Atomically increment the counter by the given value"""
        with open(self.filename, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                content = f.read().strip()
                current = int(content) if content else 0
                current += increment
                f.seek(0)
                f.write(str(current))
                f.truncate()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
            return current
    
    def get(self):
        """Get the current value of the counter"""
        with open(self.filename, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                content = f.read().strip()
                current = int(content) if content else 0
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
            return current


def setup_file_counter(tmp_dir="/tmp"):
    """
    Set up a distributed file counter.
    
    Args:
        tmp_dir: Directory to store the counter file
        
    Returns:
        tuple: (file_counter, counter_filename)
    """
    if dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            counter_filename = f"{tmp_dir}/global_rollout_counter_{uuid.uuid4().hex}.txt"
        else:
            counter_filename = ""
        filename_list = [counter_filename]
        dist.broadcast_object_list(filename_list, src=0)
        counter_filename = filename_list[0]
        file_counter = FileGlobalCounter(counter_filename)
        if rank == 0:
            file_counter.reset(0)
        dist.barrier()
    else:
        counter_filename = f"{tmp_dir}/global_rollout_counter_{uuid.uuid4().hex}.txt"
        file_counter = FileGlobalCounter(counter_filename)
        file_counter.reset(0)
    
    return file_counter, counter_filename


def reset_global_counter(file_counter):
    """Reset the global counter to 0"""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            file_counter.reset(0)
        dist.barrier()
    else:
        file_counter.reset(0)


def cleanup_counter(counter_filename):
    """Clean up the counter file"""
    if os.path.exists(counter_filename):
        try:
            os.remove(counter_filename)
        except:
            pass 