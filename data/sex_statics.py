import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 添加父目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_data import load_data
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def sex_statics(df):
    """計算性別欄位的描述性統計"""
    