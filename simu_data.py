import numpy as np
from online_utils import get_batch_data_realtime_train,get_batch_data_real_time_predict

def simulate_data_for_realtime_train_or_predict(shape,file_name):
    #[64,2498+249]
    data = np.random.randint(0,900,shape)
    with open(file_name, "w", newline="") as csv_file:
        for item in data:
            item_str = ",".join(str(i) for i in item)
            csv_file.write(item_str + "\n")


if __name__=='__main__':
    # 配置文件
    # simulate_data_for_realtime_train_or_predict([128,4487+249],file_name = "hdfs/simulate_data_for_realtime_train.csv")
    # simulate_data_for_realtime_train_or_predict([1, 4487],
    #                                        file_name="hdfs/simulate_data_for_realtime_predict.csv")
    date, traffic_input, predicts = get_batch_data_realtime_train('hdfs/simulate_data_for_realtime_train.csv')
    date_p, traffic_input_p = get_batch_data_real_time_predict("hdfs/simulate_data_for_realtime_predict.csv")
    print("hah")
