import tvm
import tvm.testing
from tvm import te
import numpy
import timeit
import sys
import argparse


# TVM Matrix Multiplication using TE
def define_gemm(M, N, K, dtA, dtB, dtC):
   
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A", dtype=dtA)
    #B = te.placeholder((N, K), name="B", dtype=dtB)
    B = te.placeholder((K, N), name="B", dtype=dtB)

    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * 
        B[k, y], axis=k), name="C")
        #B[y, k], axis=k), name="C")

    return A, B, C

def schedule_gemm_v2(A, B, C, tgt_gpu, elements_per_thread, num_thread, vt, step):
    
    s = te.create_schedule(C.op)


    # Establecemos las opciones de tiling
    elements_per_thread = elements_per_thread
    num_thread = num_thread
    block_factor = elements_per_thread * num_thread
    step = step
    vthread = vt

    # Obtenemos los indices de thread de GPU
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = te.thread_axis((0, num_thread), "threadIdx.y")

    # Dividimos las cargas de trabajo
    mi, ni = s[C].op.axis
    bx, mi = s[C].split(mi, factor=block_factor)
    by, ni = s[C].split(ni, factor=block_factor)

    # Asignamos las variables de iteracion a los indices de thread de GPU
    s[C].bind(bx, block_x)
    s[C].bind(by, block_y)

    tx, mi = s[C].split(mi, nparts=num_thread)
    ty, ni = s[C].split(ni, nparts=num_thread)
    
    s[C].reorder(bx, by, tx, ty, mi, ni)
    s[C].unroll(mi)
    s[C].vectorize(ni)

    s[C].bind(tx, thread_x)
    s[C].bind(ty, thread_y)

    
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):

        f = tvm.lower(s, [A, B, C], name="gpu_gemm", simple_mode=False)

        func = tvm.build(s, [A, B, C], target=tgt_gpu, name="mmult")

        assert func
    
    return func

def schedule_gemm_v1(A, B, C, tgt_gpu, elements_per_thread, num_thread, vt, step):
    
    s = te.create_schedule(C.op)

    # Establecemos las opciones de tiling
    elements_per_thread = elements_per_thread
    num_thread = num_thread
    block_factor = elements_per_thread * num_thread
    step = step
    vthread = vt

    # Obtenemos los indices de thread de GPU
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = te.thread_axis((0, num_thread), "threadIdx.y")

    # Dividimos las cargas de trabajo
    mi, ni = s[C].op.axis
    by, mi = s[C].split(mi, factor=block_factor)
    bx, ni = s[C].split(ni, factor=block_factor)

    # Asignamos las variables de iteracion a los indices de thread de GPU
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)

    ty, mi = s[C].split(mi, nparts=num_thread)
    tx, ni = s[C].split(ni, nparts=num_thread)
    
    #s[C].reorder(by, bx, tyz, txz, ty, tx, mi, ni)
    s[C].reorder(by, bx, ty, tx, mi, ni)
    s[C].unroll(mi)
    s[C].vectorize(ni)

    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)

    
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):

        f = tvm.lower(s, [A, B, C], name="gpu_gemm", simple_mode=False)

        func = tvm.build(s, [A, B, C], target=tgt_gpu, name="mmult")

        assert func
    
    return func

def schedule_gemm_v3(A, B, C, tgt_gpu, elements_per_thread, num_thread, vt, step):
    
    s = te.create_schedule(C.op)


    # Establecemos las opciones de tiling
    elements_per_thread = elements_per_thread
    num_thread = num_thread
    block_factor = elements_per_thread * num_thread
    step = step
    vthread = vt

    # Obtenemos los indices de thread de GPU
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = te.thread_axis((0, num_thread), "threadIdx.y")

    # Dividimos las cargas de trabajo
    mi, ni = s[C].op.axis
    by, mi = s[C].split(mi, factor=block_factor)
    bx, ni = s[C].split(ni, factor=block_factor)

    # Asignamos las variables de iteracion a los indices de thread de GPU
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)

    ty, mi = s[C].split(mi, nparts=num_thread)
    tx, ni = s[C].split(ni, nparts=num_thread)
    
    #s[C].reorder(by, bx, tyz, txz, ty, tx, mi, ni)
    s[C].reorder(by, bx, ty, tx, mi, ni)
    s[C].unroll(mi)
    s[C].vectorize(ni)

    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)

    
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):

        f = tvm.lower(s, [A, B, C], name="gpu_gemm", simple_mode=False)

        func = tvm.build(s, [A, B, C], target=tgt_gpu, name="mmult")

        assert func
    
    return func


def schedule_gemm_v4(A, B, C, tgt_gpu, elements_per_thread, num_thread, vt, step):
    
    s = te.create_schedule(C.op)


    CL = s.cache_write(C, "local")

    # Establecemos las opciones de tiling
    elements_per_thread = elements_per_thread
    num_thread = num_thread
    block_factor = elements_per_thread * num_thread
    step = step
    vthread = vt

    # Obtenemos los indices de thread de GPU
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = te.thread_axis((0, num_thread), "threadIdx.y")

    # Dividimos las cargas de trabajo
    mi, ni = s[C].op.axis
    by, mi = s[C].split(mi, factor=block_factor)
    bx, ni = s[C].split(ni, factor=block_factor)

    # Asignamos las variables de iteracion a los indices de thread de GPU
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)

    ty, mi = s[C].split(mi, nparts=num_thread)
    tx, ni = s[C].split(ni, nparts=num_thread)
    
    #s[C].reorder(by, bx, tyz, txz, ty, tx, mi, ni)
    s[C].reorder(by, bx, ty, tx, mi, ni)
    s[C].unroll(mi)
    s[C].vectorize(ni)

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

    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):

        f = tvm.lower(s, [A, B, C], name="gpu_gemm", simple_mode=False)

        func = tvm.build(s, [A, B, C], target=tgt_gpu, name="mmult")

        assert func
    
    return func

def schedule_gemm(A, B, C, batch, tgt_gpu, elements_per_thread, num_thread, vt, step):
    
    s = te.create_schedule(C.op)
    """
    if batch != 1:
        AA = s.cache_read(A, "shared", [C])
        BB = s.cache_read(B, "shared", [C])
    #AL = s.cache_read(AA, "local", [C])
    #BL = s.cache_read(BB, "local", [C])
    """
    CL = s.cache_write(C, "local")
   
    # Establecemos las opciones de tiling
    elements_per_thread = elements_per_thread
    num_thread = num_thread
    block_factor = elements_per_thread * num_thread
    step = step
    vthread = vt

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
    #s[C].reorder(by, bx, ty, tx, mi, ni)
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
    """
    if batch != 1:
        # Unimos el computo a las variables de iteracion
        s[AA].compute_at(s[CL], rko)
        s[BB].compute_at(s[CL], rko)
        #s[AL].compute_at(s[CL], rki)
        #s[BL].compute_at(s[CL], rki)


        # Schedule para la carga en la memoria compartida AA
        mi, ni = s[AA].op.axis
        ty, mi = s[AA].split(mi, nparts=num_thread)
        tx, ni = s[AA].split(ni, nparts=num_thread)
        _, ni = s[AA].split(ni, factor=4)
        s[AA].reorder(ty, tx, mi, ni)
        s[AA].bind(ty, thread_y)
        s[AA].bind(tx, thread_x)
        s[AA].vectorize(ni)  # vectorize memory load
#
        # Schedule para la carga en la memoria compartida BB
        mi, ni = s[BB].op.axis
        ty, mi = s[BB].split(mi, nparts=num_thread)
        tx, ni = s[BB].split(ni, nparts=num_thread)
        _, ni = s[BB].split(ni, factor=4)
        s[BB].reorder(ty, tx, mi, ni)
        s[BB].bind(ty, thread_y)
        s[BB].bind(tx, thread_x)
        s[BB].vectorize(ni)  # vectorize memory load
    """
    
    # Optimizamos AL y BL
    #mi, ni = s[AL].op.axis
    #xo, mi = s[AL].split(mi, factor=4)
    #s[AL].unroll(xo)
    #s[AL].vectorize(mi)
    #mi, ni = s[BL].op.axis
    #s[BL].vectorize(ni)
    
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):

        f = tvm.lower(s, [A, B, C], name="gpu_gemm", simple_mode=False)

        #print(f)
        #print(pp)

        func = tvm.build(s, [A, B, C], target=tgt_gpu, name="mmult")

        assert func
    
    return func

def perf_eval(d_a, d_b, d_c, func, tgt, dev):
    
    evaluator = func.time_evaluator(func.entry_name, dev, number=100,  min_repeat_ms=500)
    mean_time = evaluator(d_a, d_b, d_c).mean

    return mean_time

def check_gemm(M, N, K, gpu_gemm, dtA, dtB, dtC, tgt_gpu):

    target = tvm.target.Target(target="llvm", host="llvm")
    dev = tvm.device(target.kind.name, 0)

    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtA), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtB), dev)

    np_repeat = 10

    np_running_time = timeit.timeit(
            setup="import numpy\n"
            "M = " + str(M) + "\n"            
            "K = " + str(K) + "\n"
            "N = " + str(N) + "\n"
            "dtype = "+str(dtC) +"\n"
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
    d_c = tvm.nd.array(numpy.zeros((M, N), dtype=dtC), dev)
    
    gpu_gemm(d_a, d_b, d_c)
    
    return tvm.testing.assert_allclose(d_c.numpy(), answer, rtol=1e-5)


def test_gemm_gpu(M, N, K, batch, dtA, dtB, dtC, qnn, check, device="cuda"):

    M = M * batch

    if device == "cuda":
        tgt_gpu = tvm.target.Target(target=device, host="llvm")
    else:
        tgt_gpu = tvm.target.cuda(model='unknown', arch=device, options=None)
    
    
    dev = tvm.device(tgt_gpu.kind.name, 0)

    best_time = float('inf')
    best_ept = 0 #elements per thread
    best_nt = 0 # number of threas
    best_s = 0 # steps
    best_vt = 0
    
    nt_values=[4,8,16,32]
    ept_values=[1,2,4,8,16,32]
    vt_values=[1,2,4,8]
    s_values=[1,2,4,8,16,32]
    
    if check == True:
        make_check = 0
    dev = tvm.device(tgt_gpu.kind.name, 0)
    d_b = tvm.nd.array(numpy.random.rand(K, N).astype(dtB), dev)
    d_a = tvm.nd.array(numpy.random.rand(M, K).astype(dtA), dev)
    d_c = tvm.nd.array(numpy.zeros((M, N), dtype=dtC), dev)
    
    A,B,C = define_gemm(M ,N, K, dtA, dtB, dtC)
    
    for num_thread in nt_values:
        for elements_per_thread in ept_values:
            for vt in vt_values:
                for step in s_values:
                    if vt > elements_per_thread:
                        print("%i, %i, %i %i is not suitable for %i %i %i" % (num_thread,elements_per_thread, vt, step, M, N, K))
                        continue
                    
                    try:
                        gpu_gemm = schedule_gemm(A, B, C,batch,tgt_gpu, elements_per_thread, num_thread, vt, step)
                    except:
                        print("%i, %i, %i %i is not suitable for %i %i %i" % (num_thread,elements_per_thread, vt, step, M, N, K))
                        continue 
                    try:
                        if check == True and make_check == 0:
                            make_check = 1
                            if check_gemm(M,N,K,gpu_gemm, dtA, dtB, dtC, tgt_gpu) == False:
                                print("ERROR");
                    except:
                        print("ERROR during check\n")
                        break
                    
                    try:
                        time = perf_eval(d_a, d_b, d_c, gpu_gemm, tgt_gpu, dev)
                        print("{} {} {} {} Time: {}".format(num_thread, elements_per_thread, vt, step, time))
                        if time <= best_time:
                            best_time=time
                            best_ept=elements_per_thread
                            best_nt = num_thread
                            best_s = step
                            best_vt = vt
                    except:
                        print("ERROR during evaluation\n")
                        break
                    
                    continue
                continue
            continue
        continue

    print("{} {} {} {} {}".format("M", "N", "K", "time", "gflops"))
    print("{} {} {} ({} {} {} {}) {} {}".format(M, N, K, best_nt,best_ept, best_vt, best_s, best_time, ((2*M*N*K)/(best_time))/1e9))

