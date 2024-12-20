import sys
import argparse


from gemm_gpu_tvm import *
from autogemm_gpu_tvm import *
#from quantize import *
from models import *


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
    for M, N, K in MNK:
        M = M * args.batch
        if(args.auto == False):
            test_gemm_gpu(M, N, K, args.dtA, args.dtB, args.dtC, args.qnn, args.check, args.arch)
        else:
            auto_gemm_gpu(M, N, K, args.dtA, args.dtB, args.dtC, args.qnn, args.check, args.arch)



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



