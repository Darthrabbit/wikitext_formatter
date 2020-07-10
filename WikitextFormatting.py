import wikitextparser as wtp
import nltk
import pandas as pd


class WikitextChainFormatter:
    """
    Format Wikipedia Texts and extract chain via different criteria from them
    """

    def __init__(self, pathToFile, criteria, stopWords, exceptions, minArticleLength):
        self.pathToFile = pathToFile
        self.stopWords = stopWords
        self.exceptions = exceptions
        self.minArticleLength = minArticleLength
        
        parsed = wtp.parse(self.read_file(pathToFile))
        labels, articles = self.split_into_titles_and_articles(parsed)

        labels = [self.clean_titles(label, exceptions, self.stopWords) for label in labels]
        articles = [self.clean_articel(article, exceptions, self.stopWords, self.minArticleLength) for article in
                    articles]

        self.data = pd.DataFrame({"Labels": labels, "Articel": articles})
        self.frequency = self.get_frequency_from_data()

    def read_file(self, path):
        with open(path, "r") as fp:
            content = fp.readlines()

        for i in range(len(content)):
            content[i] = content[i].strip()
            content[i] += '\n'

        content = "".join(content)
        return content

    def split_into_titles_and_articles(self, parsed):
        def get_header_level(header):
            return int(header.count('=') / 2 + 1)

        labels = []
        articels = []
        new_articel = []
        is_new = False

        labels.append(parsed.sections[1].title)
        new_articel.append(parsed.sections[1].contents)

        for sec in parsed.sections[2:]:
            if get_header_level(sec.title) == 1:
                labels.append(sec.title)
                articels.append("".join(new_articel))
                new_articel = []

            new_articel.append(sec.contents)

        articels.append("".join(new_articel))

        return labels, articels

    def remove_punctuation(articel, exceptions):
        words = articel.split(" ")
        words = [word for word in words if (word.isalpha() or word in exceptions)]
        cleaned = " ".join(words)

        return cleaned

    def stop_word_removal(self, articel, stop_words):
        words = articel.split(" ")
        words = [word for word in words if word not in stop_words]
        cleaned = " ".join(words)

        return cleaned

    def clean_articel(self, articel, exceptions, stop_words, min_length):
        sentences = nltk.sent_tokenize(articel)
        sentences = [sent.lower() for sent in sentences]
        sentences = [sent.strip() for sent in sentences]
        sentences = [sent for sent in sentences if len(sent) >= min_length]
        sentences = [self.stop_word_removal(sent, stop_words) for sent in sentences]
        sentences = [self.remove_punctuation(sent, exceptions) for sent in sentences]
        sentences = [sent + "\n" for sent in sentences if len(sent) > min_length]

        return "".join(sentences)

    def clean_titles(self, title, exceptions, stop_words):
        cleaned_title = title.lower().strip()
        cleaned_title = self.stop_word_removal(cleaned_title, stop_words)
        cleaned_title = self.remove_punctuation(cleaned_title, exceptions)

        return "".join(cleaned_title)

    def get_frequency_from_data(self):
        return nltk.FreqDist(nltk.word_tokenize(" ".join([" ".join(self.data[frame]) for frame in self.data])))


class WikitextCriteria:
    """
    Use to define what kind of words should be chosen from a given text
    """
