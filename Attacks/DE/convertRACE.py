# convert RACE to Squad format dataset
import json
import os
import glob
import random
random.seed(2020)

train_race = ['RACE/train/high', 'RACE/train/middle']
dev_race = ['RACE/dev/high', 'RACE/dev/middle']
test_race = ['RACE/test/high', 'RACE/test/middle']
out_dir = 'spanRACE/'
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

def convert_race(paths, taskname):
	## paths is a list containing all data folders
	examples = []
	squad_data = {}
	squad_data['version'] = 'spanRACE'
	squad_data['data'] = []
	squad_file = out_dir+taskname+'spanRace.json'
	counter = 0
	with open(squad_file, 'w') as f:
		for path in paths:
			filenames = glob.glob(path+"/*txt")
			for filename in filenames:
				with open(filename, 'r', encoding='utf-8') as fpr:
					data_raw = json.load(fpr)
					txt_id = data_raw['id']
					article = data_raw['article']
					entry = {}
					# one entry refers to one RACE file
					entry['title'] = txt_id
					entry['paragraphs'] = []
					# one para refers to one PQA triple 
					## for each question
					## note that each question has a different passage in this case
					for i in range(len(data_raw['answers'])):
						counter += 1
						PQA = {}
						truth = ord(data_raw['answers'][i]) - ord('A')
						question = data_raw['questions'][i]
						options = data_raw['options'][i]
						answer = options[truth]

						# insert options into passage 
						# don't change test passages!!
						# do squad eval on dev set 
						# generate n_best on test set!
						if taskname != 'test_':
							article_tok = article.split()
							for opt in options:
								article_tok.insert(random.randrange(0, len(article_tok)), opt)
							new_article = ' '.join(article_tok)
							answer_start = new_article.find(answer)
							if answer_start == -1:
								print ("Skip one example during : ", filename)
								continue
						else:
							# test set
							new_article = article
							answer_start = 0 # no need ans pos 

						# save into json file
						PQA['context'] = new_article
						QA = {}
						QA['question'] = question
						QA['id'] = filename + '-' + str(i)
						QA['answers'] = [{'text': answer,
										  'answer_start': answer_start}]
						PQA['qas'] = [QA]
						# ignore is_impossible, all are possible
						entry['paragraphs'].append(PQA)
					squad_data['data'].append(entry)

		json.dump(squad_data, f, indent=4)



# convert_race(train_race, 'train_') #-> train_spanRace.json
# convert_race(dev_race, 'dev_')  #->	dev_spanRace.json
convert_race(test_race, 'test_') #-> test_spanRace.json





