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
        labels, articles = self.split_into_titles_and_articles(parsed)

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


class WikitextCriteria:
    """
    Use to define what kind of words should be chosen from a given text
    """
