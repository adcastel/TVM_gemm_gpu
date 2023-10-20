import numpy as np
import tvm
import tvm.testing
from tvm import te
import time

def tvm_min_max_par(m,n,nt=8,et=1,vt=2,cmin=True,tgt="cuda"):
    
    A = te.placeholder((m, n), name="A")
    k = te.reduce_axis((0, n), name="k")
    k2 = te.reduce_axis((0, m), name="k2")
    # there are two way to use this min reducer:
    # mode 1, accept (expr, axis, where) to produce an Reduce Expr
    # tvm.min represents tvm.te.min or tvm.tir.min.
    one=1
    if cmin == True:
        B = te.compute((m,), lambda i: te.min(A[i, k], axis=k), name="B")
        C = te.compute((one,0), 
            lambda i: 
            te.min(B[k2], axis=k2), 
            name="C")
    else:
        B = te.compute((m,), lambda i: te.max(A[i, k], axis=k), name="B")
        C = te.compute((one,0), 
            lambda i: 
            te.max(B[k2], axis=k2), 
            name="C")

    s = te.create_schedule(C.op)
    
    bfactor= nt * et
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis((0, nt), "threadIdx.x")
    vthread_x = te.thread_axis((0, vt), "vthread", name="vx")
    
    
    (mi,) = s[B].op.axis
    by, mi = s[B].split(mi, factor=bfactor)
    s[B].bind(by, block_x)
    
    vty, mi = s[B].split(mi, nparts=vt)
    ty, mi = s[B].split(mi, nparts=nt)

    s[B].vectorize(mi)

    s[B].bind(vty,vthread_x)
    s[B].bind(ty,thread_x)
    
    block_xb = te.thread_axis("blockIdx.x")
    thread_xb = te.thread_axis((0, nt), "threadIdx.x")
    vthread_xb = te.thread_axis((0, vt), "vthread", name="vx")
    (mi,) = s[C].op.axis
    by2, mi = s[C].split(mi, factor=bfactor)
    s[C].bind(by2, block_xb)
    
    tgt_gpu = tvm.target.Target(target="cuda", host="llvm")
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        f = tvm.lower(s, [A,C], name="gpu_quant", simple_mode=False)
        #print(f)
        func = tvm.build(s, [A,C], target=tgt_gpu, name="quant")
        assert func
    return func

"""
def tvm_quantize_(m,n):
    #m = te.var("m")
    
    A = te.placeholder((m, n), name="A")
    k = te.reduce_axis((0, n), name="k")
    k2 = te.reduce_axis((0, m), name="k2")
    # there are two way to use this min reducer:
    # mode 1, accept (expr, axis, where) to produce an Reduce Expr
    # tvm.min represents tvm.te.min or tvm.tir.min.
    Bmin = te.compute((m,), lambda i: te.min(A[i, k], axis=k), name="Bmin")
    Bmax = te.compute((m,), lambda i: te.max(A[i, k], axis=k), name="Bmax")
    print(Bmin.op)
    print(Bmax.op)
    nn=1
    Cmin = te.compute((nn,0), 
            lambda i: 
            te.min(Bmin[k2], axis=k2), 
            name="Cmin")
    print(Cmin.op)
    Cmax = te.compute((nn,0), 
            lambda i: 
            te.min(Bmax[k2], axis=k2), 
            name="Cmax")
    print(Cmax.op)
    nnn=2
    cero=0
    C = te.compute((nnn,0),
            lambda i:
            te.if_then_else(i<1,Cmin[0],Cmax[0]), name="C")
            
    print(C.op)
    s = te.create_schedule(C.op)
    nt=8
    
    block_x = te.thread_axis("blockIdx.x")
    #block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis((0, nt), "threadIdx.x")
    #thread_y = te.thread_axis((0, 1), "threadIdx.y")
    #print(s[B].op.axis)
    (mi,) = s[Bmin].op.axis
    by, mi = s[Bmin].split(mi, factor=nt)
    #print(block_x, thread_x, mi)
    s[Bmin].bind(by, block_x)
    
    (mi,) = s[Bmax].op.axis
    by, mi = s[Bmax].split(mi, factor=nt)
    #print(block_x, thread_x, mi)
    s[Bmax].bind(by, block_x)
    
    
    nt=8
    block_xb = te.thread_axis("blockIdx.x")
    thread_xb = te.thread_axis((0, nt), "threadIdx.x")
    
    (mi,) = s[Cmin].op.axis
    by2, mi = s[Cmin].split(mi, factor=nt)
    s[Cmin].bind(by2, block_xb)
    
    block_xbm = te.thread_axis("blockIdx.x")
    thread_xbm = te.thread_axis((0, nt), "threadIdx.x")
    (mi,) = s[Cmax].op.axis
    by2m, mi = s[Cmax].split(mi, factor=nt)
    s[Cmax].bind(by2m, block_xbm)
   
    block_xbc = te.thread_axis("blockIdx.x")
    thread_xbc = te.thread_axis((0, 1), "threadIdx.x")
    (mi,) = s[C].op.axis
    by2, mi = s[C].split(mi, factor=1)
    s[C].bind(by2, block_xbc)

    tgt_gpu = tvm.target.Target(target="cuda", host="llvm")
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        f = tvm.lower(s, [A,Bmin,Bmax,Cmin,Cmax,C], name="gpu_quant", simple_mode=False)
        print(f)
        func = tvm.build(s, [A,Bmin,Bmax,Cmin,Cmax,C], target=tgt_gpu, name="quant")
        assert func
    return func

"""
"""
def tvm_quantize2(m,n):
    #m = te.var("m")
    
    A = te.placeholder((m, n), name="A")
    k = te.reduce_axis((0, n), name="k")
    k2 = te.reduce_axis((0, m), name="k2")

    # there are two way to use this min reducer:
    # mode 1, accept (expr, axis, where) to produce an Reduce Expr
    # tvm.min represents tvm.te.min or tvm.tir.min.
    nn=1
    C = te.compute((nn,), 
            lambda i: 
            te.min(A[k2,k], axis=[k,k2]), 
            name="C")

    s = te.create_schedule(C.op)
    
    block_xb = te.thread_axis("blockIdx.x")
    nt=8
    thread_xb = te.thread_axis((0, nt), "threadIdx.x")
    (mi,) = s[C].op.axis
    by2, mi = s[C].split(mi, factor=nt)
    s[C].bind(by2, block_xb)
    
    tgt_gpu = tvm.target.Target(target="cuda", host="llvm")
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        f = tvm.lower(s, [A,C], name="gpu_quant", simple_mode=False)
        #print(f)
        func = tvm.build(s, [A,C], target=tgt_gpu, name="quant")
        assert func
    return func
"""


def quantize(arr: np.ndarray, q_min: int = -128, q_max: int=127, dtype: np.dtype = np.int8):
    
        """Quantize the array in the given datatype range    
        Args:
        arr (np.ndarray): Input array in fp32/fp16
        dtype (np.dtype): datatype of output array. Defaults to np.int8
        q_min (int, optional): minimum value in the data type. Defaults to -128.
        q_max (int, optional): maximum value in the data type. Defaults to 127.
        
        """
        # 1 
        min_val, max_val = np.min(arr), np.max(arr)
        print (min_val)
        print (max_val)
        # 2
        max_val = np.maximum(np.abs(min_val), max_val)
        # 3
        scale = max_val / ((q_max - q_min) / 2)
        print(scale)
        #4
        out = np.round(arr / scale).astype(np.int8)
        
        return out


def compute_quantization(m,n, nt=8, et=2, vt=2, st=1):

    A = te.placeholder((m, n), name="A")
    scale = te.placeholder((1,),name="scale")


    out = te.compute(
            (m,n),
            lambda i, j:
            tvm.topi.cast(te.round(te.div(A[i,j],scale[0])),'int8'),
            name = "out"
            )
    
    s = te.create_schedule(out.op)

    num_thread=nt
    elements_per_thread=et
    vthread=vt
    block_factor= nt * et
    # Obtenemos los indices de thread de GPU
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = te.thread_axis((0, vthread), "vthread", name="vx")
    thread_yz = te.thread_axis((0, vthread), "vthread", name="vy")
 
    # Dividimos las cargas de trabajo
    mi, ni = s[out].op.axis
    by, mi = s[out].split(mi, factor=block_factor)
    bx, ni = s[out].split(ni, factor=block_factor)
    
    # Asignamos las variables de iteracion a los indices de thread de GPU
    s[out].bind(by, block_y)
    s[out].bind(bx, block_x)

    tyz, mi = s[out].split(mi, nparts=vthread)  # virtual thread split
    txz, ni = s[out].split(ni, nparts=vthread)  # virtual thread split
    ty, mi = s[out].split(mi, nparts=num_thread)
    tx, ni = s[out].split(ni, nparts=num_thread)
    
    s[out].reorder(by, bx, tyz, txz, ty, tx, mi, ni)
    s[out].unroll(mi)
    s[out].vectorize(ni)

    s[out].bind(tyz, thread_yz)
    s[out].bind(txz, thread_xz)
    s[out].bind(ty, thread_y)
    s[out].bind(tx, thread_x)

    # Schedule de la memoria de escritura local CL
    AA = s.cache_read(A, "shared", [out])
    AL = s.cache_read(AA, "local", [out])
    #CL = s.cache_write(out, "local")
    
    #s[CL].compute_at(s[out], tx)
    #mi, ni = s[CL].op.axis
    #rk, = s[CL].op.reduce_axis
    #rko, rki = s[CL].split(rk, factor=step)
    #s[CL].reorder(rko, rki, mi, ni)
    #s[CL].unroll(mi)
    #xo, ni = s[CL].split(ni, factor=4)
    #s[CL].unroll(xo)
    #s[CL].vectorize(ni)
    
    
    # Unimos el computo a las variables de iteracion
    
    s[AA].compute_at(s[out], ty)
    s[AL].compute_at(s[out], tx)

    # Optimizamos AL y BL
    mi, ni = s[AL].op.axis
    xo, mi = s[AL].split(mi, factor=4)
    s[AL].unroll(xo)
    s[AL].vectorize(mi)
    
    # Schedule para la carga en la memoria compartida AA
    mi, ni = s[AA].op.axis
    ty, mi = s[AA].split(mi, nparts=num_thread)
    tx, ni = s[AA].split(ni, nparts=num_thread)
    _, ni = s[AA].split(ni, factor=4)
    s[AA].reorder(ty, tx, mi, ni)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    s[AA].vectorize(ni)  # vectorize memory load
    
    tgt_gpu = tvm.target.Target(target="cuda", host="llvm")
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        f = tvm.lower(s, [A,scale,out], name="gpu_quant", simple_mode=False)
        #print(f)
        func = tvm.build(s, [A,scale,out], target=tgt_gpu, name="quant")
        assert func
    
    return func

def generate_functions(M,K, opt=True):
    
    bf_min= None #tvm_min_max_par(M,K, 8, 1, 2, cmin=True)
    #print("min")
    bf_max= None #tvm_min_max_par(M,K, 8, 1, 2, cmin=False)
    #print("max")
    bf_out=None

    tgt_gpu = tvm.target.Target(target="cuda", host="llvm")
    dev = tvm.device(tgt_gpu.kind.name, 0)
    d_a = tvm.nd.array(np.random.rand(M, K).astype("float32"), dev)
    d_c = tvm.nd.array(np.zeros((M,K), dtype="int8"), dev)
    d_mn = tvm.nd.array(np.zeros((1,), dtype="float32"), dev)
    scale = tvm.nd.array(np.random.rand(1, ).astype("float32"), dev)


    if opt:
        
        nt_values=[2,4,8,16,32]
        ept_values=[1,2,4,8]
        vt_values=[1,2]
        
        nt_values=[32]
        #ept_values=[8]
        #vt_values=[2]
        
        ttn= float('inf')
        ttx= float('inf')
        ttq= float('inf')
        bntx ,betx,bvtx = 0, 0, 0
        bntm ,betm,bvtm = 0, 0, 0
        bntq ,betq,bvtq = 0, 0, 0
        for nt in nt_values:
            for et in ept_values:
                for vt in vt_values:

                    #MIN
                    try:
                        func_min = tvm_min_max_par(M,K, nt, et, vt, cmin=True)
                        func_min(d_a,d_mn)
                        evaluator = func_min.time_evaluator(func_min.entry_name, dev, number=1000, min_repeat_ms=500)
                        time = evaluator(d_a, d_mn).mean
                        print("MIN {} {} {} {} {} {}".format(M,K,nt,et,vt,time))
                        if time < ttn:
                            bntm ,betm,bvtm = nt, et, vt
                            ttn = time
                            bf_min = func_min
                            bf_max = tvm_min_max_par(M,K, nt, et, vt, cmin=False)
                    #        print("--->BEST MIN and MAX:{}  {} {} {} {} {}".format(ttn, M,K, nt, et, vt,))
                    except:
                        print("MIN does not support {} {} {}".format(nt ,et,vt))

                    # MAX
                    
                    #try:
                   #     func_max = tvm_min_max_par(M,K, nt, et, vt, cmin=False)
                   #     func_max(d_a,d_mn)
                   #     evaluator = func_max.time_evaluator(func_max.entry_name, dev, number=1000, min_repeat_ms=500)
                   #     time = evaluator(d_a, d_mn).mean
                   #     print("MAX {} {} {} {} {} {}".format(M,K,nt,et,vt,time))
                   #     if time < ttx:
                   #         bntx ,betx,bvtx = nt, et, vt
                   #         ttx = time
                   #         bf_max = func_max
                   # 
                   # except:
                   #     print("MAX does not support {} {} {}".format(nt ,et,vt))
                    # QUANTIZE
                    #try:
                    #    func_out = compute_quantization(M,K, nt, et, vt)
                    #    func_out(d_a,scale,d_c)
                    #    evaluator = func_out.time_evaluator(func_out.entry_name, dev, number=1000, min_repeat_ms=500)
                    #    time = evaluator(d_a, scale, d_c).mean
                    #    print("QUANT {} {} {} {} {} {}".format(M,K,nt,et,vt,time))
                    #    if time < ttq:
                    #        bntq ,betq,bvtq = nt, et, vt
                    #        ttq = time
                    #        bf_out = func_out
                    ##        print("--->BEST QUANT: {} {} {} {} {} {}".format(ttq, M,K, nt, et, vt,))
                    #except:
                    #    print("QUANTIZE does not support {} {} {}".format(nt ,et,vt))

            print("\n")
        print("Optimization finished:")
    
        print("Time MIN: {} nt: {} et: {} vt: {}".format(ttn,bntm ,betm,bvtm))
        print("Time MAX: {} nt: {} et: {} vt: {}".format(ttn,bntm ,betm,bvtm))
        #print("Time max: {} nt: {} et: {} vt: {}".format(ttx, bntx ,betx,bvtx))
        print("Time quant: {} nt: {} et: {} vt: {}".format(ttq, bntq ,betq,bvtq))
    else:
        bf_min= tvm_min_max_par(M,K, 32, 1, 1, cmin=True)
        bf_max= tvm_min_max_par(M,K, 32, 1, 1, cmin=False)
        bf_out = compute_quantization(M,K,8,2,1)
        

    evaluator = bf_min.time_evaluator(bf_min.entry_name, dev, number=1000, min_repeat_ms=500)
    tmin = evaluator(d_a, d_mn).mean
    print("Time fmin = {}".format(tmin)) 
    evaluator = bf_max.time_evaluator(bf_max.entry_name, dev, number=1000, min_repeat_ms=500)
    tmax = evaluator(d_a, d_mn).mean
    print("Time fmax = {}".format(tmax)) 
    bf_out = compute_quantization(M,K)
    evaluator = bf_out.time_evaluator(bf_out.entry_name, dev, number=1000, min_repeat_ms=500)
    print("Time quanti {}".format(evaluator(d_a, scale,d_c).mean))

    return bf_min, bf_max, bf_out

def myquantization(A,C,fmin,fmax,fout,dev, q_min: int = -128, q_max: int=127):
    d_mn = tvm.nd.array(np.zeros((1,), dtype="float32"), dev)
    d_mx = tvm.nd.array(np.zeros((1,), dtype="float32"), dev)
    fmin(A,d_mn)
    fmax(A,d_mx)
    print(d_mn, d_mx)
    max_val = np.maximum(np.abs(d_mn.numpy()), d_mx.numpy())
    scal = max_val / ((q_max - q_min) / 2)
    #print("Antes A\n", A)
    scale =  tvm.nd.array(scal, dev)
    print(scale)
    fout(A,scale,C)
    #print("Despues A\n", C)



M=2000
K=2000

fmin, fmax, fout = generate_functions(M,K, opt=False)

tgt_gpu = tvm.target.Target(target="cuda", host="llvm")
dev = tvm.device(tgt_gpu.kind.name, 0)

a = np.random.rand(M, K).astype("float32")
#print(a)
ts = time.time()
b = quantize(a)
te = time.time()
#print("Referencia ",b)
print("Total time np = {}".format(te-ts))

d_a = tvm.nd.array(a, dev)
d_c = tvm.nd.array(np.zeros((M,K), dtype="int8"), dev)
ts = time.time()
myquantization(d_a,d_c,fmin,fmax,fout,dev, -128, 127)
te = time.time()
print("Total time GPU = {}".format(te-ts))
#tvm.testing.assert_allclose(d_c.numpy(), b, rtol=1e-5)
print("OK?", np.allclose(d_c.numpy(), b, rtol=1e-5))


"""
d_a = tvm.nd.array(a, dev)
d_c = tvm.nd.array(np.zeros((M,K), dtype="int8"), dev)
d_b1 = tvm.nd.array(np.zeros((M,), dtype="float32"), dev)
d_b2 = tvm.nd.array(np.zeros((M,), dtype="float32"), dev)
d_c = tvm.nd.array(np.zeros((1,), dtype="float32"), dev)
func(d_a,d_b1,d_b2,d_c)
print("Bmin despues", d_b1)
print("Bmax despues", d_b2)
print("C despues", d_c)
"""
"""
func = tvm_min_max_par(M,K, cmin=True)


tgt_gpu = tvm.target.Target(target="cuda", host="llvm")
dev = tvm.device(tgt_gpu.kind.name, 0)
#d_a = tvm.nd.array(np.random.rand(M, K).astype("float32"), dev)

a = np.random.rand(M, K).astype("float32")
print("Referencia ",quantize(a))

d_a = tvm.nd.array(a, dev)
"""
"""
d_bmin = tvm.nd.array(np.zeros((M,), dtype="float32"), dev)
d_bmax = tvm.nd.array(np.zeros((M,), dtype="float32"), dev)
d_cmin = tvm.nd.array(np.zeros((1,), dtype="float32"), dev)
d_cmax = tvm.nd.array(np.zeros((1,), dtype="float32"), dev)
"""
"""
d_c = tvm.nd.array(np.zeros((1,), dtype="float32"), dev)
d_cx = tvm.nd.array(np.zeros((1,), dtype="float32"), dev)

print("antes", d_c)
func(d_a,d_c)
print("C despues", d_c)
func2 = tvm_min_max_par(M,K, cmin=False)
func2(d_a,d_cx)
#print("B despues", d_b)
print("C despues", d_cx)

evaluator = func.time_evaluator(func.entry_name, dev, number=1000)
mean_time1 = evaluator(d_a, d_c).mean
"""
"""
func = tvm_quantize2(M,K)

tgt_gpu = tvm.target.Target(target="cuda", host="llvm")
dev = tvm.device(tgt_gpu.kind.name, 0)
d_c2 = tvm.nd.array(np.zeros((1,), dtype="float32"), dev)

#print("antes", d_a)
func(d_a,d_c2)
#print("B despues", d_b)
print("C despues", d_c2)

evaluator = func.time_evaluator(func.entry_name, dev, number=1000)
mean_time2 = evaluator(d_a, d_c2).mean
"""



#print("M {} N {} time: {}".format(M,K,mean_time1))
