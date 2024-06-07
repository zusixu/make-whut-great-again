import argparse
import numpy as np

class Env_parameters():
    def parse_args(self):
        parser = argparse.ArgumentParser('Env_parameters')

        # 定义模型参数
        parser.add_argument('--a', type=float, default=1.0, help='example for parameters setting')

        return parser.parse_args()
