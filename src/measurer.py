from codecarbon import EmissionsTracker
import psutil
from src import utils
import tracemalloc
import platform
import torch
import pandas as pd
import os
import time
import sys
import numpy as np


# noinspection PyAttributeOutsideInit
class Measurer:

    def __int__(self):
        # DONE
        self.data_size_in_grid_points = ''
        # TO DO
        self.largest_allocated_array_in_grid_points = ''
        # DONE
        self.data_size = 0
        # DONE
        self.main_memory_available = 0
        # DONE
        self.main_memory_consumed = 0
        # TO DO
        self.sum_of_allocated_variable_sizes = 0
        # DONE
        self.cpu_gpu_description = ''
        # DONE
        self.wall_time = 0
        # DONE
        self.max_energy_consumed = 0
        # DONE
        self.co2_consumed = 0
        # DONE
        self.network_traffic = 0
        # TO DO
        self.storage_cost = 0
        self.compute_cost = 0
        self.network_cost = 0
        # DONE
        self.programming_language = 'Python'
        # DONE
        self.essential_libraries = ''

    def start_compute_wall_time(self):
        self.wall_time = time.time()
        return time

    def end_compute_wall_time(self):
        t = self.wall_time
        end = time.time()
        self.wall_time = end - t
        return end

    def start_compute_data_size(self, data_path, aws_s3, aws_session):
        if aws_s3:
            s3 = aws_session.resource('s3')
            bucket = s3.Bucket(data_path)
            size_in_bytes = 0
            for key in bucket.objects.all():
                size_in_bytes += key.size
            self.data_size = size_in_bytes
        else:
            self.data_size = psutil.disk_usage(data_path).used

    def end_compute_data_size(self, data_path, aws_s3, aws_session):
        start = self.data_size
        if aws_s3:
            s3 = aws_session.resource('s3')
            bucket = s3.Bucket(data_path)
            size_in_bytes = 0
            for key in bucket.objects.all():
                size_in_bytes += key.size
            end = size_in_bytes
        else:
            end = psutil.disk_usage(data_path).used
        end = end - start
        self.data_size = end
        return end

    def start_compute_main_memory_consumed(self):
        self.main_memory_consumed = 0
        tracemalloc.start()

    def end_compute_main_memory_consumed(self):
        used = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        self.main_memory_consumed = used
        return used

    def total_main_memory_available(self):
        total = psutil.virtual_memory().total
        self.main_memory_available = total
        return total

    def cpu_gpu_description(self):
        description = ''
        # Machine type
        description = description + f"Machine type: {platform.machine()}\n"
        # Processor type
        description = description + f"Processor type: {platform.processor()}\n"
        # Physical cores
        description = description + f"Number of physical cores: {psutil.cpu_count(logical=False)}\n"
        # Logical cores
        description = description + f"Number of logical cores: {psutil.cpu_count(logical=True)}\n"
        # Min frequency
        description = description + f"Min CPU frequency: {psutil.cpu_freq().min} GHz\n"
        # Max frequency
        description = description + f"Max CPU frequency: {psutil.cpu_freq().max} GHz\n"
        # GPU
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            description = description + 'CUDNN VERSION:', torch.backends.cudnn.version() + "\n"
            description = description + 'Number CUDA Devices:', torch.cuda.device_count() + "\n"
            description = description + 'CUDA Device Name:', torch.cuda.get_device_name(0) + "\n"
            description = description + 'CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(
                0).total_memory / 1e9 + "\n"
        else:
            description = description + 'No GPU available\n'
        self.cpu_gpu_description = description
        return description

    def start_compute_co2_emissions(self):
        self.co2_consumed = 0
        # energy_consumed is in kW, Emissions as CO₂-equivalents [CO₂eq], in kg
        tracker = EmissionsTracker(log_level='error', measure_power_secs=1)
        tracker.start()
        return tracker

    def end_compute_co2_emissions(self, tracker):
        emissions = tracker.stop()
        self.co2_consumed = emissions
        return emissions

    def compute_data_size_in_grid_points(self, shape):
        if len(shape) == 0:
            self.data_size_in_grid_points = ''
            return ''
        data_size_grid_points = str(shape[0]) + "x" + str(shape[1])
        self.data_size_in_grid_points = data_size_grid_points
        return data_size_grid_points

    def compute_energy_consumed(self):
        emissions = pd.read_csv('emissions.csv')
        kw = emissions['energy_consumed'].iloc[-1]
        self.max_energy_consumed = kw
        os.remove('emissions.csv')
        return kw

    def start_compute_network_traffic(self):
        sent = psutil.net_io_counters().bytes_sent
        recv = psutil.net_io_counters().bytes_recv
        self.network_traffic = sent + recv

    def end_compute_network_traffic(self):
        sent = psutil.net_io_counters().bytes_sent
        recv = psutil.net_io_counters().bytes_recv
        new_value = sent + recv
        old_value = self.network_traffic
        self.network_traffic = new_value - old_value

    def get_essential_libraries(self, libraries, program_path):
        f = open(program_path, 'r')
        lib = libraries
        while True:
            code_line = f.readline()
            if not code_line:
                break
            if code_line[0] == '#':
                continue
            if not (code_line[:5] == 'from ' and code_line.__contains__('import')):
                continue
            code_line = code_line.split(' ')[1]
            if code_line == 'types' or code_line == 'measurer':
                continue
            if code_line.__contains__('.'):
                code_line = code_line.split('.')[0]
            if code_line not in lib:
                lib.append(code_line)
        lib_str = ''
        for i in lib:
            lib_str = lib_str + i + "\n"
        self.essential_libraries = lib_str
        return lib_str

    def write_out(self, csv_file):
        csv = pd.DataFrame(columns=['Measure', 'Value'])
        csv.loc[len(csv)] = {'Measure': 'Data size in grid points ', 'Value': self.data_size_in_grid_points}
        csv.loc[len(csv)] = {'Measure': 'Largest allocated array in grid points ',
                             'Value': self.largest_allocated_array_in_grid_points}
        csv.loc[len(csv)] = {'Measure': 'Data size (MB)',
                             'Value': str(utils.bytes_to(self.data_size, 'm'))}
        csv.loc[len(csv)] = {'Measure': 'Main memory available (GB)',
                             'Value': str(utils.bytes_to(self.main_memory_available, 'g'))}
        csv.loc[len(csv)] = {'Measure': 'Main memory consumed (GB)',
                             'Value': str(utils.bytes_to(self.main_memory_consumed, 'g'))}
        csv.loc[len(csv)] = {'Measure': 'Sum of allocated variable sizes (GB)',
                             'Value': str(utils.bytes_to(self.sum_of_allocated_variable_sizes, 'g'))}
        csv.loc[len(csv)] = {'Measure': 'Description of CPU/GPU', 'Value': self.cpu_gpu_description}
        csv.loc[len(csv)] = {'Measure': 'Wall time in seconds', 'Value': self.wall_time}
        csv.loc[len(csv)] = {'Measure': 'Energy consumed (kw)', 'Value': self.max_energy_consumed}
        csv.loc[len(csv)] = {'Measure': 'CO₂-equivalents [CO₂eq] (kg)', 'Value': self.co2_consumed}
        csv.loc[len(csv)] = {'Measure': 'Network traffic (MB)',
                             'Value': str(utils.bytes_to(self.network_traffic, 'm'))}
        # csv.loc[len(csv)] = {'Measure': 'Storage cost', 'Value': self.storage_cost}
        # csv.loc[len(csv)] = {'Measure': 'Compute cost', 'Value': self.compute_cost}
        # csv.loc[len(csv)] = {'Measure': 'Network cost', 'Value': self.network_cost}
        csv.loc[len(csv)] = {'Measure': 'Programming language', 'Value': 'Python'}
        csv.loc[len(csv)] = {'Measure': 'Essential libraries', 'Value': self.essential_libraries}
        csv.to_csv(csv_file, index=False)
        return csv

    def compute_variable_sizes(self, variables):
        variables_sum = 0
        largest_array_size = 0
        largest_array_shape = ''
        for key in variables:
            variables_sum = variables_sum + sys.getsizeof(variables[key])
            if isinstance(variables[key], np.ndarray) or isinstance(variables[key], pd.DataFrame):
                if variables[key].size > largest_array_size:
                    largest_array_size = variables[key].size
                    largest_array_shape = str(variables[key].shape).replace('(', '[').replace(')', ']')
        self.sum_of_allocated_variable_sizes = variables_sum
        self.largest_allocated_array_in_grid_points = largest_array_shape

    # start
    def start(self, data_path='/', logger=None, aws_s3=False, aws_session=None):
        self.start_compute_wall_time()
        if logger is not None:
            logger.info(
                "Started computational costs meter: wall time, memory consumed, network traffic, CO2 emissions, data size"
            )
        self.start_compute_main_memory_consumed()
        self.start_compute_network_traffic()
        tracker = self.start_compute_co2_emissions()
        self.start_compute_data_size(data_path, aws_s3, aws_session)
        return tracker

    # end
    def end(self, tracker, shape, libraries, csv_file, variables, data_path='/', program_path=__file__, logger=None, aws_s3=False,
            aws_session=None):
        self.end_compute_main_memory_consumed()
        self.end_compute_co2_emissions(tracker)
        self.end_compute_network_traffic()
        self.compute_data_size_in_grid_points(shape)
        self.compute_energy_consumed()
        self.total_main_memory_available()
        self.cpu_gpu_description()
        self.get_essential_libraries(libraries, program_path)
        self.end_compute_data_size(data_path, aws_s3, aws_session)
        self.end_compute_wall_time()
        self.compute_variable_sizes(variables)
        csv = self.write_out(csv_file)
        if logger is not None:
            logger.info("Stopped computational costs meter. Results saved at" + csv_file)
        print(csv)
