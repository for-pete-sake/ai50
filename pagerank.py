import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    key ={}
    p = round(float((1-damping_factor)/len(corpus.keys())),5)
    
    for i in corpus.keys():
        key[i] = p

    links = corpus[page]
    n = len(links)

    if n==0:
        for i in corpus.keys():
            key[i] += (damping_factor/len(corpus.keys()))
        return key


    for i in links:
        key[i] += (damping_factor/n)
    return key


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    key= {}
    r = random.randint(0,len(corpus.keys())-1)
    a = list(corpus.keys())[r]
    count = 0
    for i in corpus.keys():
        key[i] = 0

    while count<n:
        key[a] +=1
        count +=1
        rand = random.random() ##Ceck and possible change this##
        dic = transition_model(corpus,a, damping_factor)
        for i in dic:
            if dic[i] <rand:
                rand -= dic[i]
            else:
                a = i
                break
    
    normalize = sum(key.values())
    for i in key.keys():
        key[i] = round(key[i]/normalize, 5)
    return key



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pr = {}
    for i in corpus.keys():
        pr[i] = float(1/len(corpus.keys()))

    stop = 0
    while not stop:
        topr = {}

        stop = 1
        for i in pr.keys():
            temp = pr[i]

            topr[i] = float((1-damping_factor)/len(corpus.keys()))

            for page, link, in corpus.items():
                if i in link:
                    topr[i]+= float(damping_factor*pr[page]/len(link))

            if abs(temp-topr[i] > 0.001):
                stop = 0

        for i in pr.keys():
            pr[i] = topr[i]

    return pr


if __name__ == "__main__":
    main()
