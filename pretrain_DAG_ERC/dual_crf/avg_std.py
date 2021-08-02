import numpy as np
f = open('uar.txt')
num_list = []
for line in f.readlines():
    #print(float(line))
    num_list.append(float(line))

arr = np.asarray(num_list)

print(arr.shape)
print(np.std(arr, ddof=1))
print(np.mean(arr))
f.close()