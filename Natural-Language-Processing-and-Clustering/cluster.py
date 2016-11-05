import sys


import numpy
from nltk.cluster import KMeansClusterer, GAAClusterer, euclidean_distance
import nltk.corpus
import nltk.stem

stemmer_func = nltk.stem.snowball.SnowballStemmer("english").stem

stopwords = set(nltk.corpus.stopwords.words("english"))

def normalize_word(word):
    return stemmer_func(word.lower())

def get_words(titles):
    words = set()
    for title in job_titles:
        for word in title.split():
            words.add(normalize_word(word))
    return list(words)


def vectorspaced(title):
    title_components = [normalize_word(word) for word in title.split()]
    return numpy.array([
        word in title_components and not word in stopwords
        for word in words], numpy.short)

title_file = open("opfile.txt", 'r')

job_titles = [line.strip() for line in title_file.readlines()]
words = get_words(job_titles)


cluster = KMeansClusterer(7, euclidean_distance)
cluster.cluster([vectorspaced(title) for title in job_titles if title])
classified_examples = [cluster.classify(vectorspaced(title)) for title in job_titles]

for cluster_id, title in sorted(zip(classified_examples, job_titles)):
    print (cluster_id, title)
