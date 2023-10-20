import sys
import argparse


from gemm_gpu_tvm import *
#from quantize import *



def main(args):
    
    print( args)
    test_gemm_gpu(args.M, args.N, args.K, args.batch,args.dtA, args.dtB, args.dtC, args.qnn, args.check, args.arch)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TVM matmul generator')
    parser.add_argument('--M', type=int, default=1000, help='M dimension')
    parser.add_argument('--N', type=int, default=1000, help='N dimension')
    parser.add_argument('--K', type=int, default=1000, help='K dimension')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--dtA', type=str, default="float32", help='data type A')
    parser.add_argument('--dtB', type=str, default="float32",  help='data type B')
    parser.add_argument('--dtC', type=str, default="float32", help='data type C')
    parser.add_argument('--arch', type=str, default="cuda", help='Device compute capability')
    parser.add_argument('--qnn', action=argparse.BooleanOptionalAction, default=False, help='quantize')
    parser.add_argument('--check', action=argparse.BooleanOptionalAction, default=False, help='check?')
    args = parser.parse_args()
    
    main(args)



