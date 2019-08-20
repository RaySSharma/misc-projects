def condense(one, two):
	for i in range(1, min(len(one), len(two))):
	    if one[-i:] == two[:i]:
	        return [''.join(one[:-i]) + ''.join(two)]
	return [one, two]

sentence = input('Input Sentence: ').split(' ')
for i in range(len(sentence)-1):
	try:
		sentence[i:i+2] = condense(sentence[i], sentence[i+1])
	except:
		break
print(' '.join(sentence))