


import sys
import random

for i in range(10000):
    caseid = random.randint(100000, 999999)
    label = random.randint(10, 1000)
    id_list = []
    for j in range(random.randint(5,30)):
        id_list.append(str(random.randint(0, 10000)))
    st = str(caseid) + "\t" + str(label) + "\t" + ",".join(id_list)
    print st
