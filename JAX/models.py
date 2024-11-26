def test():
    return [[100,100,100]]

def square():
    return [
            [1024,1024,1024],
            [1024*2,1024*2,1024*2],
            [1024*3,1024*3,1024*3],
            [1024*4,1024*4,1024*4],
            [1024*5,1024*5,1024*5],
            [1024*6,1024*6,1024*6],
            [1024*7,1024*7,1024*7],
            [1024*8,1024*8,1024*8],
            [1024*9,1024*9,1024*9],
            [1024*10,1024*10,1024*10],
            ]

def square2():
    return [
            [1000,1000,1000],
            [2000,2000,2000],
            [3000,3000,3000],
            [4000,4000,4000],
            [5000,5000,5000],
            [6000,6000,6000],
            [7000,7000,7000],
            [8000,8000,8000],
            [9000,9000,9000],
            [10000,10000,10000]
            ]

def resnet():
    return [
            [12544, 64, 147], 
            [3136, 64, 64],
            [3136, 64, 576], 
            [3136, 256, 64],
            [3136, 64, 256],
            [3136, 128, 256],
            [784, 128, 1152],
            [784, 512, 128],
            [784, 512, 256],
            [784, 128, 512],
            [784, 256, 512],
            [196, 256, 2304],
            [196, 1024, 256],
            [196, 1024, 512], 
            [196, 256, 1024],
            [196, 512, 1024], 
            [49, 512, 4608],  
            [49, 2048, 512], 
            [49, 2048, 1024],
            [49, 512, 2048]]

def googlenet(): 
    return [[50176, 192, 27],
            [50176, 64, 192],
            [50176, 96, 192],
            [50176, 128, 864],
            [50176, 16, 192],
            [50176, 32, 144],
            [50176, 32, 288],
            [50176, 32, 192],
            [50176, 128, 256],
            [50176, 192, 1152],
            [50176, 32, 256],
            [50176, 96, 288],
            [50176, 96, 864],
            [50176, 64, 256],
            [12544, 192, 480],
            [12544, 96, 480],
            [12544, 208, 864],
            [12544, 16, 480],
            [12544, 48, 144],
            [12544, 48, 432],
            [12544, 64, 480],
            [12544, 160, 512],
            [12544, 112, 512],
            [12544, 224, 1008],
            [12544, 24, 512],
            [12544, 64, 216],
            [12544, 64, 576],
            [12544, 64, 512],
            [12544, 128, 512],
            [12544, 256, 1152],
            [12544, 144, 512],
            [12544, 288, 1296],
            [12544, 32, 512],
            [12544, 64, 288],
            [12544, 256, 528],
            [12544, 160, 528],
            [12544, 320, 1440],
            [12544, 32, 528],
            [12544, 128, 288],
            [12544, 128, 1152],
            [12544, 128, 528],
            [3136, 256, 832],
            [3136, 160, 832],
            [3136, 320, 1440],
            [3136, 32, 832],
            [3136, 128, 288],
            [3136, 128, 1152],
            [3136, 128, 832],
            [3136, 384, 832],          
            [3136, 192, 832],
            [3136, 384, 1728],
            [3136, 48, 832],
            [3136, 128, 432]]
