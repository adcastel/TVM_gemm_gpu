from time import time
from models import *
import os
import sys
import argparse
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np

import skcuda.linalg as culinalg
import skcuda.misc as cumisc


def gemm_culib(M,N,K,dtA,dtB,dtC,qnn,check,arch):
    
    culinalg.init()

    A = np.random.randn(M,K)
    B = np.random.randn(K,N)
    C = np.random.randn(M,N)
    
    A_gpu = gpuarray.to_gpu(A)
    B_gpu = gpuarray.to_gpu(B)
    C_gpu = gpuarray.to_gpu(C)
    
    #C_d = cublas.gemm("N", "N", 1.0, A_d, B_d)
    C_gpu = culinalg.dot(A_gpu, B_gpu)
    repeticiones = 10000

    start = time()
    for _ in range (repeticiones):
        C_gpu = culinalg.dot(A_gpu, B_gpu)
    end = time()
    tt = (end-start)/repeticiones
    gflops=2.0*M*N*K/tt/1.0e9
    print(f"{M} {N} {K} {gflops}")
    del A_gpu
    del B_gpu
    del C_gpu
def main(args):
    
    print( args)
    
    if args.model=='square':
        MNK=square()
        args.batch=1
    elif args.model == 'resnet':
        MNK = resnet()
    elif args.model == 'googlenet':
        MNK = googlenet()
    else:
        MNK = test()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    for M, N, K in MNK:
        M = M * args.batch
        gemm_culib(M, N, K, args.dtA, args.dtB, args.dtC, args.qnn, args.check, args.arch)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TVM matmul generator')
    #parser.add_argument('--M', type=int, default=1000, help='M dimension')
    #parser.add_argument('--N', type=int, default=1000, help='N dimension')
    #parser.add_argument('--K', type=int, default=1000, help='K dimension')
    parser.add_argument('--model', type=str, default="test", help='name of model')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--dtA', type=str, default="float32", help='data type A')
    parser.add_argument('--dtB', type=str, default="float32",  help='data type B')
    parser.add_argument('--dtC', type=str, default="float32", help='data type C')
    parser.add_argument('--arch', type=str, default="cuda", help='Device compute capability')
    parser.add_argument('--qnn', action=argparse.BooleanOptionalAction, default=False, help='quantize')
    parser.add_argument('--check', action=argparse.BooleanOptionalAction, default=False, help='check?')
    parser.add_argument('--auto', action=argparse.BooleanOptionalAction, default=False, help='check?')
    args = parser.parse_args()
    
    
    main(args)




