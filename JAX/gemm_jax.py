import jax.numpy as jnp
from jax.lib import xla_bridge
import jax
import sys
import argparse
from jax import random
from time import time
from models import *
import os
import timeit

@jax.jit
def jit_gemm(A,B):
    return jnp.matmul(A, B)
    #return jnp.dot(A,B)

def gemm_jax(M,N,K,dtA,dtB,dtC,qnn,check,arch):
    jax.config.update('jax_default_matmul_precision', 'highest')
    key = random.PRNGKey(0)
    A_jax = random.uniform ( key , shape =( M , K ) )
    B_jax = random.uniform ( key , shape =( K , N ) )
    C_jax = random.uniform ( key , shape =( M , N ) )
    #A_jax = jax.device_put(A)
    #B_jax = jax.device_put(B)
    #C_jax = jax.device_put(C)
    
#    # code snippet to be executed only once
#    mysetup = '''
#import jax; 
#import jax.numpy as jnp; 
#from jax.lib import xla_bridge
#from jax import random
##@jax.jit
#def jit_gemm(A,B):
#    return jnp.matmul(A, B)
#    #return jnp.dot(A,B)
#jax.config.update('jax_default_matmul_precision', 'highest')
#key = random.PRNGKey(0)
#M=5000
#N=5000
#K=5000
#A = random.uniform ( key , shape =( M , K ) )
#B = random.uniform ( key , shape =( K , N ) )
#C = random.uniform ( key , shape =( M , N ) )
#A_jax = jax.device_put(A)
#B_jax = jax.device_put(B)
#C_jax = jax.device_put(C)
#'''
#
#    # code snippet whose execution time is to be measured
#    mycode = '''
#C_jax = jit_gemm(A_jax,B_jax)
#(jax.device_put(0.) + 0).block_until_ready()
#'''
#
#    # timeit statement
#    nreps=1000
#    print(timeit.timeit(setup=mysetup, stmt=mycode,number=1))
#    tt = timeit.timeit(setup=mysetup, stmt=mycode,number=nreps)
#    tt=tt/nreps
#    flops=2*5000*5000*5000/1e9
#    print(tt,flops/tt)
#    
#    
    C_jax = jit_gemm(A_jax,B_jax)
    (jax.device_put(0.) + 0).block_until_ready()
    C_jax = jit_gemm(A_jax,B_jax)
    (jax.device_put(0.) + 0).block_until_ready()
    repeticiones = 1000

    start = time()
    for _ in range (repeticiones):
        C_jax = jit_gemm(A_jax,B_jax)
        (jax.device_put(0.) + 0).block_until_ready()
    end = time()
    tt = (end-start)/repeticiones
    gflops=2.0*M*N*K/tt/1.0e9
    del A_jax
    del B_jax
    del C_jax
    print(f"{M} {N} {K} {tt} {gflops}")

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
    devices = jax.local_devices()
    print(jax.devices())
    
    for M, N, K in MNK:
        M = M * args.batch
        gemm_jax(M, N, K, args.dtA, args.dtB, args.dtC, args.qnn, args.check, args.arch)

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




