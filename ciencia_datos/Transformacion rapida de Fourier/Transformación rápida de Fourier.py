from math import sin,cos,pi

# Recursive function of FFT
def fft(a):

    n = len(a)

    # if input contains just one element
    if n == 1:
        return [a[0]]

    # For storing n complex nth roots of unity
    theta = -2*pi/n
    w = list( complex(cos(theta*i), sin(theta*i)) for i in range(n) ) 
    
    # Separe coefficients
    Aeven = a[0::2]
    Aodd  = a[1::2]

    # Recursive call for even indexed coefficients
    Yeven = fft(Aeven) 

    # Recursive call for odd indexed coefficients
    Yodd = fft(Aodd)

    # for storing values of y0, y1, y2, ..., yn-1.
    Y = [0]*n
   
    middle = n//2
    for k in range(n//2):
        w_yodd_k  = w[k] * Yodd[k]
        yeven_k   =  Yeven[k]
        
        Y[k]          =  yeven_k  +  w_yodd_k
        Y[k + middle] =  yeven_k  -  w_yodd_k
    
    return Y


# Driver code
if __name__ == '__main__':

    a = [1, 2, 3, 4]
    b = fft(a)
    for B in b:
        print(B)