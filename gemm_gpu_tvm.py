import tvm

import tvm.testing

from tvm import te

import numpy

import timeit

import sys


# The size of the matrix

# (M, K) x (K, N)

# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.

M = int(sys.argv[1])

N = int(sys.argv[2])

K = int(sys.argv[3])



# The default tensor data type in tvm

dtype = "float32"



# You will want to adjust the target to match any CPU vector extensions you

# might have. For example, if you're using using Intel AVX2 (Advanced Vector

# Extensions) ISA for SIMD, you can get the best performance by changing the

# following line to ``llvm -mcpu=core-avx2``, or specific type of CPU you use.

# Recall that you're using llvm, you can get this information from the command

# ``llc --version`` to get the CPU type, and you can check ``/proc/cpuinfo``

# for additional extensions that your processor might support.



target = tvm.target.Target(target="llvm", host="llvm")

dev = tvm.device(target.kind.name, 0)



# Random generated tensor for testing

a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)

b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)



# Repeatedly perform a matrix multiplication to get a performance baseline

# for the default numpy implementation

np_repeat = 10

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

#--------------------------------------------------------------------------------------------


# TVM Matrix Multiplication using TE

tgt_gpu = tvm.target.Target(target="cuda", host="llvm")

k = te.reduce_axis((0, K), "k")

A = te.placeholder((M, K), name="A")

B = te.placeholder((K, N), name="B")

C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

#print(type(C))



def vfinal(elements_per_thread, num_thread, step):

        # Designamos la jerarquia de memoria
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



def evaluate_operation(s, vars, target, name, optimization, log):

    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):

        f = tvm.lower(s, [A, B, C], name="b3a2c0", simple_mode=False)

        #print(f)

        func = tvm.build(s, [A, B, C], target=tgt_gpu, name="mmult")

        assert func

    assert func

    dev = tvm.device(tgt_gpu.kind.name, 0)

    d_a = tvm.nd.array(a, dev)

    d_b = tvm.nd.array(b, dev)

    d_c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)

    func(d_a, d_b, d_c)

    tvm.testing.assert_allclose(d_c.numpy(), answer, rtol=1e-5)


    evaluator = func.time_evaluator(func.entry_name, dev, number=100)

    mean_time = evaluator(d_a, d_b, d_c).mean

    print("%s: %f" % (optimization, mean_time))

    log.append((optimization, mean_time))

    return mean_time



# Tiempo y parametros optimos
best_time = float('inf')
best_i = 0
best_j = 0
best_k = 0

# Valores para los 3 parametros
rango_i=[2,4,8,16]
rango_j=[4,8,16,32]
rango_k=[1,2,4,8,16,32,64]

# Construir y probar los distintos schedule
for i in rango_i:
    for j in rango_j:
        for k in rango_k:
            try:
                # Creamos el schedule
                s = te.create_schedule(C.op)

                vfinal(i,j,k)

                with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):

                    f = tvm.lower(s, [A, B, C], name="b3a2c0", simple_mode=False)

                    #print(f)

                    func = tvm.build(s, [A, B, C], target=tgt_gpu, name="mmult")

                    assert func



                dev = tvm.device(tgt_gpu.kind.name, 0)


                d_a = tvm.nd.array(a, dev)

                d_b = tvm.nd.array(b, dev)

                d_c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)


                func(d_a, d_b, d_c)

                tvm.testing.assert_allclose(d_c.numpy(), answer, rtol=1e-5)



                log = []
                print("%i, %i, %i" % (i, j, k))
                mean_time=evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="none", log=log)

                if(mean_time<=best_time):
                    best_time=mean_time
                    best_i=i
                    best_j=j
                    best_k=k
                    print("NUEVO OPTIMO")

            except:
                print("%i, %i, %i no es una combinacion valida para este tamanyo de problema" % (i, j, k))

# Resultados
print("COMBINACION OPTIMA: elements_per_thread=%i, num_threads=%i, step=%i" % (best_i, best_j, best_k))
print("TIEMPO MEDIO DE EJECUCION: %f" % (best_time))

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