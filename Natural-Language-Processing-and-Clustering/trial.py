import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy
from nltk.cluster import KMeansClusterer, GAAClusterer, euclidean_distance


file = open("example_jobs.txt", 'r').read()

stop_words = set(stopwords.words("english"))
sf = open("sentence_file.txt","w")
sentences = sent_tokenize(file)
#print (sentences)
sf.write(str(sentences))

o = open("opfile.txt","w")
stop = open("stop.txt", "w")

for i in word_tokenize(file):
	o.write(i + "\n")

filter = []
words = word_tokenize(file)
for w in words:
	if w not in stop_words:
		filter.append(w)
stop.write(str(filter) + "\n")
#print (filter)

ps = PorterStemmer()
for w in filter:
	stemm = ps.stem(w)
	o.write(stemm + "\n")
	#print(stemm)



	

