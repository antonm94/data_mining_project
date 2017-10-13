import numpy as np


shingles_file = open('data/handout_shingles.txt')
shingles_list = shingles_file.readlines()
shingles = np.zeros((700, 100), dtype='uint16')


for i in range(len(shingles_list)):  # len is 700
    line = shingles_list[i].split()
    shingles[i] = line[1:]
print(shingles)
