import sys
import os
import xml.etree.ElementTree
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

inverted_index = {}
nltk.download('stopwords')
nltk.download('punkt')
tokenizer = RegexpTokenizer(r'\w+')
stopwords = set(stopwords.words("english"))
ps = PorterStemmer()

def extract_words_from_file(file_name):
    tree = xml.etree.ElementTree.parse(file_name)
    root = tree.getroot()
    for doc in root.findall("RECORD"):
        text = extract_words_from_doc(doc)


def extract_words_from_doc(doc):
    doc_id, doc_text = "", ""
    for elem in doc:
        if elem.tag == "RECORDNUM":
            doc_id = elem.text
        elif elem.tag == "TITLE" or elem.tag == "EXTRACT" or elem.tag == "ABSTRACT":
            text = str(elem.text)
            doc_text += text

    text_tokens = tokenizer.tokenize(doc_text)
    filtered_text = filter_stop_words(text_tokens)
    stemmed_text = words_stemming(filtered_text)
    doc_dict = create_doc_dict(stemmed_text)
    update_inverted_index(doc_id, doc_dict)

def filter_stop_words(text_tokens):
    return [w for w in text_tokens if not w.lower() in stopwords]


def words_stemming(filtered_text):
    return [ps.stem(w) for w in filtered_text]

def create_doc_dict(stemmed_text):
    dict = {}
    for word in stemmed_text:
        if word not in dict:
            dict[word] = 1
        else:
            dict[word] += 1
    return dict


def update_inverted_index(doc_id, doc_dict):
    max_word_num = doc_dict[max(doc_dict, key=doc_dict.get)]
    for word in doc_dict:
        if word not in inverted_index:
            inverted_index[word] = {"df" : 1, "list" : {doc_id : doc_dict[word]/max_word_num}} #the value of doc_id is tf score
        else:
            inverted_index[word]["df"] += 1
            linked_list = inverted_index[word]["list"]
            linked_list[doc_id] = doc_dict[word]/max_word_num


if __name__ == "__main__":
    if sys.argv[1] == "create_index":
        path = sys.argv[2]
        for filename in os.listdir(path):
            if filename.endswith("xml"):
                f = os.path.join(path,filename)
                extract_words_from_file(f)
        x = 1



