import argparse
import numpy as np

class Env_parameters():
    def parse_args(self):
        parser = argparse.ArgumentParser('Env_parameters')

        # 定义模型参数
        parser.add_argument('--data_path', default='data/C1.xlsx',
                            help='Data file path')
        parser.add_argument('--figure_path', default='figure',
                            help='Figure file path')
        parser.add_argument('--interval', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            help='interval')
        parser.add_argument('--interval_segment', type=list, default=[(0, 287), (293, 592), (592, 856), (860, 1018), (1022, 1251), (1279, 1685), (1724, 1882), (1892, 2071), (2078, 2515), (2539, 2734), (2738, 2919)], help='Interval of each segment')

        parser.add_argument('--normal_segment', default=[0, 1, 2, 4, 5, 7, 8],
                            help='Segments use for polynomial fitting')
        parser.add_argument('--abnormal_segment', default=[3, 6, 9, 10],
                            help='Segments use for computing k_mean')
        parser.add_argument('--SLD', type=int, default=30,
                            help='Size of local data for fitting k_mean model ')
        parser.add_argument('--SRW', type=int, default=30,
                            help='Size of rolling window')
        parser.add_argument('--MMI', type=int, default=200,
                            help='Minimum maintenance interval(最小维护时间间隔)')
        parser.add_argument('--MTR', type=int, default=30,
                            help='Maintenance Time Range(维护时间范围)')


        return parser.parse_args()
