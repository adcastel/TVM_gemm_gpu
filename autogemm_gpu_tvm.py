import tvm
import tvm.testing
from tvm import te, auto_scheduler, relay
import numpy
import timeit
import sys
import argparse
import math
import os

# TVM Matrix Multiplication using TE
@auto_scheduler.register_workload
def auto_gemm(M, N, K, dtA, dtB, dtC):
   
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A", dtype=dtA)
    #B = te.placeholder((N, K), name="B", dtype=dtB)
    B = te.placeholder((K, N), name="B", dtype=dtB)

    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * 
        B[k, y], axis=k), name="C")
        #B[y, k], axis=k), name="C")

    return [A, B, C]


def auto_gemm_gpu(M, N, K, typeA, typeB, typeC, qnn, check, device="cuda"):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tgt_gpu = tvm.target.cuda(model='a100', arch="sm_80")
    #tgt_gpu = tvm.target.cuda(model='unknown', arch="sm_80")

    dev = tvm.device(tgt_gpu.kind.name, 0)

    task = tvm.auto_scheduler.SearchTask(func=auto_gemm, args=(M, N, K, typeA, typeB, typeC), target=tgt_gpu)
    folder=typeA+typeB+typeC
    print("Computational DAG:")
    print(task.compute_dag)
    trials=1000
    log_file = "auto_{}_{}/matmul_{}_{}_{}.json".format(trials,folder,M,N,K)
     
    if os.path.isfile(log_file) == False:
        tune_option = auto_scheduler.TuningOptions(
                 num_measure_trials=trials,
                 measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                 verbose=10,
                 )
         
        # Run auto-tuning (search)
        task.tune(tune_option)
    
    # Apply the best schedule
    sch, args = task.apply_best(log_file)
    
    print("Lowered TIR:")
    f = tvm.lower(sch, args, name="auto_{}_{}_{}_{}".format(M,N,K,typeC),simple_mode=True)
    
    func = tvm.build(f, target=tgt_gpu, target_host='llvm')
    
    evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=1500)
    #print("AUTO", M,N,K)
    

    
    #a_tvm = tvm.nd.array(a, device=dev)
    #b_tvm = tvm.nd.array(b, device=dev)
    a_tvm = tvm.nd.array(numpy.random.rand(M, K).astype(typeA), dev)
    b_tvm = tvm.nd.array(numpy.random.rand(K, N).astype(typeB), dev)
    out_tvm = tvm.nd.array(numpy.zeros((M, N), dtype=typeC), dev) #tvm.nd.empty(out_np.shape, dtype=typeC, device=dev)
    time = evaluator(a_tvm, b_tvm, out_tvm).mean    
    #time = np.median(evaluator(a_tvm, b_tvm, out_tvm).results) #* 1000
    
    gflops = ((2.0 * M * N * K)/(1e9*1.0))/time
    print("{} {} {} {} {} gflops".format(M, N, K, time, gflops))

