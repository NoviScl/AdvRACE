import json

with open('test_dis_scpn_copy.json', 'r') as f:
	data = json.load(f)

PQA_dict = {}
for race_id, eg in data.items():
	eg_dict = {}
	eg_dict["question"] = eg["question"]
	eg_dict["context"] = eg["context"]
	eg_dict["options"] = eg["options"][:-1]
	eg_dict["label"] = eg["label"]

	PQA_dict[race_id] = eg_dict 

with open("test_dis.json", 'w') as f:
	json.dump(PQA_dict, f, indent=4)


