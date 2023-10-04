import tvm

import tvm.testing

from tvm import te

import numpy

import timeit

import sys

import argparse


# TVM Matrix Multiplication using TE
def define_gemm(M, N, K, dt):

    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A", dtype=dt)
    B = te.placeholder((K, N), name="B", dtype=dt)

    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

    return A, B, C

def schedule_gemm(A, B, C, tgt_gpu, elements_per_thread, num_thread, step):
    
    s = te.create_schedule(C.op)


    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BL = s.cache_read(BB, "local", [C])
    CL = s.cache_write(C, "local")

    # Establecemos las opciones de tiling
    elements_per_thread = elements_per_thread
    num_thread = num_thread
    block_factor = elements_per_thread * num_thread
    step = step
    vthread = 2

    # Obtenemos los indices de thread de GPU
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = te.thread_axis((0, vthread), "vthread", name="vx")
    thread_yz = te.thread_axis((0, vthread), "vthread", name="vy")

    # Dividimos las cargas de trabajo
    mi, ni = s[C].op.axis
    by, mi = s[C].split(mi, factor=block_factor)
    bx, ni = s[C].split(ni, factor=block_factor)

    # Asignamos las variables de iteracion a los indices de thread de GPU
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)

    tyz, mi = s[C].split(mi, nparts=vthread)  # virtual thread split
    txz, ni = s[C].split(ni, nparts=vthread)  # virtual thread split
    ty, mi = s[C].split(mi, nparts=num_thread)
    tx, ni = s[C].split(ni, nparts=num_thread)
    
    s[C].reorder(by, bx, tyz, txz, ty, tx, mi, ni)
    s[C].unroll(mi)
    s[C].vectorize(ni)

    s[C].bind(tyz, thread_yz)
    s[C].bind(txz, thread_xz)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)

    # Schedule de la memoria de escritura local CL
    s[CL].compute_at(s[C], tx)
    mi, ni = s[CL].op.axis
    rk, = s[CL].op.reduce_axis
    rko, rki = s[CL].split(rk, factor=step)
    
    s[CL].reorder(rko, rki, mi, ni)
    s[CL].unroll(mi)
    xo, ni = s[CL].split(ni, factor=4)
    s[CL].unroll(xo)
    s[CL].vectorize(ni)

    # Unimos el computo a las variables de iteracion
    s[AA].compute_at(s[CL], rko)
    s[BB].compute_at(s[CL], rko)
    s[AL].compute_at(s[CL], rki)
    s[BL].compute_at(s[CL], rki)

    # Optimizamos AL y BL
    mi, ni = s[AL].op.axis
    xo, mi = s[AL].split(mi, factor=4)
    s[AL].unroll(xo)
    s[AL].vectorize(mi)
    mi, ni = s[BL].op.axis
    s[BL].vectorize(ni)

    # Schedule para la carga en la memoria compartida AA
    mi, ni = s[AA].op.axis
    ty, mi = s[AA].split(mi, nparts=num_thread)
    tx, ni = s[AA].split(ni, nparts=num_thread)
    _, ni = s[AA].split(ni, factor=4)
    s[AA].reorder(ty, tx, mi, ni)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    s[AA].vectorize(ni)  # vectorize memory load

    # Schedule para la carga en la memoria compartida BB
    mi, ni = s[BB].op.axis
    ty, mi = s[BB].split(mi, nparts=num_thread)
    tx, ni = s[BB].split(ni, nparts=num_thread)
    _, ni = s[BB].split(ni, factor=4)
    s[BB].reorder(ty, tx, mi, ni)
    s[BB].bind(ty, thread_y)
    s[BB].bind(tx, thread_x)
    s[BB].vectorize(ni)  # vectorize memory load
    
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):

        f = tvm.lower(s, [A, B, C], name="gpu_gemm", simple_mode=False)

        #print(f)

        func = tvm.build(s, [A, B, C], target=tgt_gpu, name="mmult")

        assert func
    
    return func

def perf_eval(M, N, K, dtype, func, tgt):
    
    dev = tvm.device(tgt.kind.name, 0)


    d_c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    d_a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    d_b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)
    evaluator = func.time_evaluator(func.entry_name, dev, number=100)
    mean_time = evaluator(d_a, d_b, d_c).mean

    return mean_time

def check_gemm(M, N, K, gpu_gemm, tgt_gpu):

    target = tvm.target.Target(target="llvm", host="llvm")
    dev = tvm.device(target.kind.name, 0)

    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

    np_repeat = 1

    np_running_time = timeit.timeit(
            setup="import numpy\n"
            "M = " + str(M) + "\n"            
            "K = " + str(K) + "\n"
            "N = " + str(N) + "\n"
            'dtype = "float32"\n'
            "a = numpy.random.rand(M, K).astype(dtype)\n"
            "b = numpy.random.rand(K, N).astype(dtype)\n",
            stmt="answer = numpy.dot(a, b)",
            
            number=np_repeat,
    )

    print("Numpy running time: %f" % (np_running_time / np_repeat))

    answer = numpy.dot(a.numpy(), b.numpy())
    
    dev = tvm.device(tgt_gpu.kind.name, 0)

    d_a = tvm.nd.array(a, dev)
    d_b = tvm.nd.array(b, dev)
    d_c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    
    func(d_a, d_b, d_c)
    
    return tvm.testing.assert_allclose(d_c.numpy(), answer, rtol=1e-5)


def test_gemm_gpu(M, N, K, dt, check):

    A,B,C = define_gemm(M, N, K,dt)
    
    tgt_gpu = tvm.target.Target(target="cuda", host="llvm")

    best_time = float('inf')
    best_ept = 0 #elements per thread
    best_nt = 0 # number of threas
    best_s = 0 # steps

    ept_values=[2,4,8,16]
    nt_values=[4,8,16,32]
    s_values=[1,2,4,8,16,32,64]

    for elements_per_thread in ept_values:
        for num_thread in nt_values:
            for step in s_values:
                try:
                    gpu_gemm = schedule_gemm(A, B, C,tgt_gpu, elements_per_thread, num_thread, step)
                    if check == True and check_gemm(M,N,K,gpu_gemm) == False:
                        print("ERROR");
    
                    time = perf_eval(M, N, K, dt, gpu_gemm, tgt_gpu)
                    if time <= best_time:
                        best_time=time
                        best_ept=elements_per_thread
                        best_nt = num_thread
                        best_s = step
                except:
                    pass
                    #print("%i, %i, %i is not suitable for %i %i %i" % (elements_per_thread, num_thread, step, M, N, K))
    print("{} {} {} {} {}".format("M", "N", "K", "time", "gflops"))
    print("{} {} {} {} {}".format(M, N, K, best_time, ((2*M*N*K)/(best_time))/1e9))
#print(type(C))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TVM matmul generator')

    parser.add_argument('--M', type=int, help='M dimension')
    parser.add_argument('--N', type=int, help='N dimension')
    parser.add_argument('--K', type=int, help='K dimension')
    parser.add_argument('--dt', type=str, help='data type')
    parser.add_argument('--check', type=bool, help='check?')

    args = parser.parse_args()

    test_gemm_gpu(args.M, args.N, args.K, args.dt, args.check)



#print("\n------------CODIGO GENERADO------------\n")

#print(tvm.lower(s, [A, B, C], simple_mode=True))


#if (
        #tgt_gpu.kind.name == "cuda"
        #or tgt_gpu.kind.name == "rocm"
        #or tgt_gpu.kind.name.startswith("opencl")
#):
        #func = tvm.build(s, [A, B, C], target=tgt_gpu, name="mmult")
        #dev_module = func.imported_modules[0]
        #print("-----GPU code-----")
        #print(dev_module.get_source())
#else:
        #print(fadd.get_source())
