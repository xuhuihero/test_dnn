import sys
import random

if __name__ == "__main__":
   #embeding_vec
   for i in range(10000):
       embedding = []
       for j in range(64):
          embedding.append(str(round(random.uniform(-1,1), 4)))
       print ",".join(embedding)

   for i in range(64):
       trans_mat = []
       for j in range(256):
            trans_mat.append(str(round(random.uniform(-1,1), 4)))
       print ",".join(trans_mat)

   for layer in range(10):
       in_dim = 256
       out_dim = 256
       if layer == 9:
           out_dim = 1
       for i in range(in_dim):
          weight = []
          for j in range(out_dim):
              weight.append(str(round(random.uniform(-1,1), 4)))
          print ",".join(weight)
       bias = []
       for i in range(out_dim):
          bias.append(str(round(random.uniform(-1,1), 4)))
       print ",".join(bias)

