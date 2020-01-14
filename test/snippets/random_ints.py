import numpy as np
def genIntegers(n):
    A=np.arange(n, dtype=np.int)
    for m in range(n-1,0, -1):
        x=np.random.randint(0,m)
        t=A[x]
        A[x]=A[m]
        A[m]=t
    return A

if __name__ == "__main__":
    print(genIntegers(14)+2)

    