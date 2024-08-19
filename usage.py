from joblib import load
import numpy as np
model = load('Final_model.joblib')
input_feature = np.array([[ 0.57274417, -0.5       ,  1.14354803,  0.        ,  1.35938851,
       -3.73240092,  0.91201686, -1.17674174,  1.22220127,  1.21012613,
        0.35176324,  1.25149461,  3.70144781 ]])
print(model.predict(input_feature))