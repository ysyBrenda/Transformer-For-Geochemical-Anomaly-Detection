'''
This script runs the data processing.
Author: ysyBrenda
'''
import pandas as pd
import numpy as np
import dill as pickle


def main():
    '''
    pos_feature file like: 
    =========  =========  =========  =========
    column 1   column 2   column 3   column 4  ......
    =========  =========  =========  =========
       x          y         Ag         Au      ......
    =========  =========  =========  =========

    The fisrt two columns are the x,y coordinates,
    The subsequent columns are the concentration values of each element in turn.
    '''

    readdata = pd.read_csv("data/pos_feature.csv", header=None)
    data = readdata.values
    pos = data[:, 0:2]
    feature = data[:, 2:]
    L = len(pos)
    x = []
    y = []
    x_range = max(pos[:, 0]) - min(pos[:, 0])  # 83000
    y_range = max(pos[:, 1]) - min(pos[:, 1])  # 102000

    for i in range(0, L):
        # coordinates difference, scaled!
        pos_i = (pos - pos[i, :])
        pos_i[:, 0] = pos_i[:, 0] / x_range
        pos_i[:, 1] = pos_i[:, 1] / y_range
        data_i = np.append(pos_i, feature, axis=1)
        x.append(data_i)
        y.append(data_i[i, :])
    # ================save pkl file================
    pkldata = {'x': x, 'y': y}
    pickle.dump(pkldata, open('pre_data.pkl', 'wb'))
    print("[Info] Data have saved!")


if __name__ == '__main__':
    main()
