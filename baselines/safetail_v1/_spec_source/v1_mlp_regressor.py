import numpy as np
import pickle 
import random
from sklearn.preprocessing import StandardScaler


max_mem = [828.944,828.816,829.116,834.780,853.148]
cpu = [779,740,531,399,319]

# def predict(inp):
#     # inp = np.array(inp)
#     # inp = inp.reshape((1,4))
#     # latency  = [ 1,2,4,8,16]
#     # # print("gg",latency[int(inp[0][0])-1])
#     # return latency[int(inp[0][0])-1] + math.sqrt(latency[int(inp[0][0])-1])*(random.random())
#     # print("inp" , inp)
#     st_dev = [11.6, 33.85, 957.8, 1278.25, 2040.20]
#     mlp_regr_model = open('mlp_regr_model.sav', 'rb')
#     model = pickle.load(mlp_regr_model)
#     inp[3] = (inp[3] - 320*640)/(320*640)
#     inp[1] = inp[1]/16
#     inp = np.array(inp)
#     inp = inp.reshape((1,4))

#     pred = model.predict(inp)
#     # print(pred)
#     # print(np.random.normal(0,st_dev[int(inp[0][0])-1],1))
#     pred = pred[0]  + np.random.normal(0,st_dev[int(inp[0][0])-1],1)[0]
#    # print("PRED" , pred/1000)
#     return (max(0,pred))/1000


def predict(inp):
    print("inp" , inp)  
    mlp_regr_model = open('/home/aman/yolo_thread_mlp_regressor_model.pkl', 'rb')
    # print(mlp_regr_model.summary())
    mlp_regressor = pickle.load(mlp_regr_model)
    with open('/home/aman/yolo_thread_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    X_test = np.array([[inp[0],inp[1]]])
    X_test = np.array(X_test)
    st_dev = [2.7965017993200005, 6.393525858045778, 5.515155335473335, 7.100407896557774, 6.909932092510028, 7.124802705029803, 7.972889920066626, 9.872241111272556, 8.936860060306417, 12.450634470736823, 12.444299205142089, 11.830614817831743, 14.73346686870066, 17.763701566033472, 14.910366835514143, 19.22145836272576, 17.4644685302588, 19.87528975084137, 23.439168758290045, 20.457122880591005]
    pred = mlp_regressor.predict(scaler.transform(X_test))[0]
    return (pred + np.random.normal(0 , st_dev[int(inp[0])-1] ,1))/1000



# print(predict(np.array([10,320*320])))
# print(predict([2, 828.944]))
