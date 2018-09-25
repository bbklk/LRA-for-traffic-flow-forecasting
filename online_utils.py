import numpy as np
from inference.config import *

def read_from_file(file_name):
    flow_list = []
    with open(file_name, 'r') as file:
        item_list = file.readlines()
        for item in item_list:
            items = item.strip().split(',')
            flow_list.append(list(map(float,items)))
        # flow_list[]
    flow_array = np.array(flow_list)
    return flow_array

def get_batch_data_realtime_train(file_name):
    flow_array = read_from_file(file_name)
    date, traffic, predicts = np.split(flow_array, [5, 18*INPUT_SIZE+5], axis=1)

    traffic_rs = np.reshape(traffic, [-1, INPUT_SIZE, 18])
    traffic_input = traffic_rs[:, :, 0:6], traffic_rs[:, :, 6:12], traffic_rs[:, :, 12:]
    return date, traffic_input, predicts

def get_batch_data_real_time_predict(file_name):
    flow_array = read_from_file(file_name)
    date, traffic = np.split(flow_array, [5], axis=1)

    traffic_rs = np.reshape(traffic, [-1, INPUT_SIZE, 18])
    traffic_input = traffic_rs[:, :, 0:6], traffic_rs[:, :, 6:12], traffic_rs[:, :, 12:]
    return date, traffic_input
"""
def get_batch_data_real_time_predict_hdfs(hosts,dir,file_name):
    hdfs_data = read_data_from_hdfs(hosts,dir,file_name)
    flow_array = np.array(list(map(float,hdfs_data.split(','))))

    # flow_array = read_from_file(file_name)
    date, traffic = np.split(flow_array, [5], axis=1)

    traffic_rs = np.reshape(traffic, [-1, INPUT_SIZE, 18])
    traffic_input = traffic_rs[:, :, 0:6], traffic_rs[:, :, 6:12], traffic_rs[:, :, 12:]
    return date, traffic_input
"""

# def read_data_from_hdfs(hosts,dir,file_name):
#     client = HdfsClient(hosts=hosts)
#     path = dir + file_name
#     if client.exists(path):
#         with client.open(path) as f:
#             res = f.readline().decode('utf-8').strip()
#     return res