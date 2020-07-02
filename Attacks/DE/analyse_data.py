import json

## counter the number of following-types questions
counter = 0

with open('/ps2/intern/clsi/final_distractor_datasets/unilm/test_dis.json', 'r') as f:
	data = json.load(f)

for race_id, eg in data.items():
	qn = eg["question"].lower()
	if ('true' in qn) and 'not' not in qn:
		counter += 1
		print ('Question: ', qn)
		print ('Answer: ', eg["options"][int(eg["label"])])
		print ('Distractors: ')
		for opt in range(4):
			if opt != int(eg["label"]):
				print (eg["options"][opt])
		print ('\n')

print (counter)

# print (len(data))
