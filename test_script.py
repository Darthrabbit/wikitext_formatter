import WikitextFormatting
import nltk

# This will work as a testscript and a tutorial script

filepath = "../data/wikitext-2/"
filename = "wiki.train.tokens"

pathToFile = filepath + filename
criteria = []
exceptions = []
stopWords = set(nltk.corpus.stopwords.words('english')).union(set(["<unk>"]))
minArticleLength = 1

formatter = WikitextFormatting.WikitextChainFormatter(pathToFile, criteria, stopWords, exceptions, minArticleLength)
print(formatter.data.head())