import numpy as np
import matplotlib.pyplot as plt
def get_tag(tokens, tag):
    """Extracts the value inside the given tag."""
    return tokens[tokens.index('<' + tag + '>') + 1:tokens.index('</' + tag + '>')]

def count_tokens(file_name):
	train_depths = []
	train_word_counts = []
	with open("data/negotiate/" + file_name) as f:
		for line in f:
			tokens = line.strip().split()
			tokens = get_tag(tokens, 'dialogue')
			depth = 0
			for t in tokens:
				if t == 'YOU:' or t=='THEM:':
					depth += 1
			train_depths.append(depth)
			train_word_counts.append(len(tokens)-depth)
	return (train_depths, train_word_counts)

def main():
	tr_d, tr_wc = count_tokens('test.txt')
	bins = np.arange(0, 260, 5)
	fig, ax = plt.subplots()
	plt.hist(tr_wc, bins, edgecolor='black', linewidth=0.8)
	ax.set_xticks(np.arange(0, 260, 20))
	plt.title('Number of words in dialogue')
	plt.xlabel('Word Count')
	plt.ylabel('Frequency')
	# plt.show()
	plt.savefig('words_in_convo.jpg')

	bins = np.arange(2, 22, 2)
	fig, ax = plt.subplots()
	plt.hist(tr_d, bins, edgecolor='black', linewidth=0.8)
	ax.set_xticks(np.arange(2, 22, 2))
	plt.title('Number of utterances in dialogue')
	plt.xlabel('Utterance Count')
	plt.ylabel('Frequency')
	# plt.show()
	plt.savefig('dlogs_in_convo.jpg')





if __name__ == "__main__":
	main()