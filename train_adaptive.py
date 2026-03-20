#!/usr/bin/env python3
"""
自适应模型训练入口（兼容旧命令行调用）

等效于: python train.py --model_type adaptive [其余参数...]
"""
import sys
from train import main

if __name__ == '__main__':
    sys.argv.insert(1, '--model_type')
    sys.argv.insert(2, 'adaptive')
    main()
