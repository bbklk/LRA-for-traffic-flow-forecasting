import pickle
from inference.config import *
f = open('/home/likewise-open/SENSETIME/zhaopeize/PycharmProjects/senseflow-version/input/flow_scaler.pkl', 'rb')
data = pickle.load(f)
print (data)