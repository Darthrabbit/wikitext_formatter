import wikitextparser as wtp
import nltk
import pandas as pd


class WikitextChainFormatter:
    """
    Format Wikipedia Texts and extract chain via different criteria from them
    """

    def __init__(self, pathToFile, criteria, stopWords, exceptions, minArticleLength):
        self.pathToFile = pathToFile

        parsed = wtp.parse(self.read_file(pathToFile))

    def read_file(self, path):
        with open(path, "r") as fp:
            content = fp.readlines()

        for i in range(len(content)):
            content[i] = content[i].strip()
            content[i] += '\n'

        content = "".join(content)
        return content


class WikitextCriteria:
    """
    Use to define what kind of words should be chosen from a given text
    """
