import nltk
import sys
import os
import sys
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    
    """Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    filesmap = dict()
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        with open(path, encoding = 'utf-8') as f:
            filesmap[filename] = f.read()
        
    return filesmap


def tokenize(document):
    
    """ Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document)

    rem = []

    for i in range(len(words)):
        words[i] = words[i].lower()

        if words in nltk.corpus.stopwords.words('english'):
            rem.append(i)
            continue
    for i in rem:
        del words[i]

    return words


def compute_idfs(documents):
    """
Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = {}

    for i in documents:
        content = documents[i]
        for word in content:
            if word in words:
                continue
            else:
                count = 0
                total = 0

                for temp in documents:
                    if word in documents[temp]:
                        count +=1
                    total += 1
                words[word] = math.log(float(total/count))
    return words


def top_files(query, files, idfs, n):
    
    """ Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf = {}
    for file in files:
        sum = 0

        for word in query:
            idf = idfs[word]
            sum += files[file].count(word)*idf
        tf_idf[file] = sum
    rank = sorted(tf_idf.keys(), key = lambda x: tf_idf[x], reverse=True)

    rank = list(rank)
    try:
        return rank[0:n+1]
    except:
        return rank


def top_sentences(query, sentences, idfs, n):
    
    """Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    idf = {}
    for sent in sentences:
        sum = 0
        words = sentences[sent]
        count = len(words)
        word_count = 0
        for word in query:
            word_count = word.count(word)
            if word in words:
                sum += idfs[word]

        idf[sent] = (sum, float(word_count/count))

    rank = sorted(idf.keys(), key = lambda x: idf[x], reverse= True)
    rank = list(rank)

    try:
        return rank[0:n+1]
    except:
        return rank


if __name__ == "__main__":
    main()