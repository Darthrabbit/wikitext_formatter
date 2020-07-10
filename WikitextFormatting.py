import wikitextparser as wtp
import nltk
import pandas as pd
from abc import ABC, abstractmethod


class WikitextChainFormatter:
    """
    Format Wikipedia Texts and extract chain via different criteria from them
    """

    def __init__(self, pathToFile, stopWords, exceptions, minArticleLength):

        parsed = wtp.parse(self.read_file(pathToFile))
        labels, articles = self.split_into_titles_and_articles(parsed)

        labels = [self.clean_titles(label, exceptions, stopWords) for label in labels]
        articles = [self.clean_article(article, exceptions, stopWords, minArticleLength) for article in
                    articles]

        self.data = pd.DataFrame({"Labels": labels, "Articles": articles})
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
        articles = []
        new_article = []
        is_new = False

        labels.append(parsed.sections[1].title)
        new_article.append(parsed.sections[1].contents)

        for sec in parsed.sections[2:]:
            if get_header_level(sec.title) == 1:
                labels.append(sec.title)
                articles.append("".join(new_article))
                new_article = []

            new_article.append(sec.contents)

        articles.append("".join(new_article))

        return labels, articles

    def remove_punctuation(self, article, exceptions):
        words = article.split(" ")
        words = [word for word in words if (word.isalpha() or word in exceptions)]
        cleaned = " ".join(words)

        return cleaned

    def stop_word_removal(self, article, stop_words):
        words = article.split(" ")
        words = [word for word in words if word not in stop_words]
        cleaned = " ".join(words)

        return cleaned

    def clean_article(self, article, exceptions, stop_words, min_length):
        sentences = nltk.sent_tokenize(article)
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

    def get_formatted_dataset_by_criteria(self, criterion):
        formatted = pd.DataFrame(
            {
                "Titles": self.data["Labels"],
                "article": [criterion.apply_criterion(article) for article in self.data["Articles"]]
            })

        return formatted


class WikitextCriteria(ABC):
    """
    Use to define what kind of words should be chosen from a given text
    """

    @abstractmethod
    def apply_criterion(self, data):
        pass


class FirstN(WikitextCriteria):
    def __init__(self, number_of_words):
        super().__init__()
        self.first_n_words = number_of_words

    def apply_criterion(self, data):
        tokens = nltk.word_tokenize(data)
        return tokens[0:self.first_n_words]


class POSSelection(WikitextCriteria):
    def __init__(self, tags):
        super().__init__()
        self.tags = tags

    def apply_criterion(self, data):
        pos_tags = nltk.pos_tag(nltk.word_tokenize(data))
        return [word for word, tag in pos_tags if tag in self.tags]
