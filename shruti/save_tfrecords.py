import data
import tfrecords_generator
import importlib

importlib.reload(data)
importlib.reload(tfrecords_generator)

from tfrecords_generator import write_data
from data import gen_fcst_norm
from memory_profiler import memory_usage
import numpy as np

import warnings

warnings.filterwarnings("ignore")

years = [2022]
# interval = 0.1

# mem = memory_usage((gen_fcst_norm, (year,)), interval=interval)
# time = np.arange(len(mem)) * interval
# mem_file = "log_memory_generating_norm.dat"

# np.savetxt(mem_file, np.array((time, mem)).T)

if __name__ == "__main__":
    for year in years:
        # mem = memory_usage((write_data, (year,)), interval=interval)
        write_data(year)
