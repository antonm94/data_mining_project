import numpy as np



def mapper(key,string,h1,dicc):

	s = string.split()
	doc_id = int(s[0][-4:])
	s = s[1:]
	n_hash1 = h1.shape[0]
	b = 55
	r = 20
	H1 = np.ones(n_hash1)*10000
	for i in range(n_hash1):
		for value in s:
			H1[i] = int(min(H1[i],((h1[i,0]*int(value)+h1[i,1]))%8193))

	H1 = H1.reshape((b,r))
	H2 = H1.sum(axis=1).astype(int)
	for i in range(H2.shape[0]):

		k = str(i)+' '+str(H2[i])
		if k not in dicc: dicc[k] = [doc_id,]
		else: dicc[k].append(doc_id)

	return dicc

def reducer(dicc):

	for e in dicc:
		if len(dicc[e]) > 1:
			print(e)
			print(dicc[e])
	return



def main():

	dicc = {}

	n_hash1 = 1100
	h1 = np.empty((n_hash1,2))
	for i in range(n_hash1):
		h1[i,0] = int(np.random.randint(1,8193))
		h1[i,1] = int(np.random.randint(0,8193))

	with open(filename) as file:
		for line in file:
			dicc = mapper(None,line,h1,dicc)

	reducer(dicc)

	

if __name__ == '__main__':

	filename = 'data/handout_shingles.txt'
	main()