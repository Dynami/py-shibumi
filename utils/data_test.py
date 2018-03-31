import unittest
import params
import utils.data as dt
import matplotlib.pyplot as plt
import numpy as np

class Test(unittest.TestCase):
    
    def testNorm_For_Future(self):
        _len = 16
        a = np.arange(1, _len, 1)
        print('a.shape', a.shape)
#         b = dt.norm_for_future(a, alpha=1)
#         print('b.shape', b.shape)
#         print(b)
        pass
    
    def testNormalize(self):
        df = dt.load_data(params.global_params['db_path'], 'C', index_col='date', from_date=20100101, to_date=20100405, limit=30)
        df = df['close']
        look_back = 20
        look_ahead = 1
        coeff = 3.0
        print(df.tail())
        data = df.values
        print('data.shape', data.shape)
         
         
        _, y_data = dt.normalize(data, look_back=look_back, look_ahead=look_ahead, alpha=coeff)
         
        tmp = dt.denormalize(y_data, data, look_back, look_ahead, coeff)
        print('denorma.shape', tmp.shape)
         
        plt.plot(data[look_back:], label='actual')
        plt.plot(tmp, label='denorm')
        plt.legend(loc='upper left')
        plt.show()
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testNormalize']
    unittest.main()