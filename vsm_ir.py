import xml.etree.ElementTree
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


inverted_index = {}
stopwords = set(stopwords.words("english"))

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

        text_tokens = word_tokenize(doc_text)
        filtered_text = filter_stop_words(text_tokens)

def filter_stop_words(text_tokens):
    return [w for w in text_tokens if not w.lower() in stopwords]

def
extract_words_from_doc("cfc-xml_corrected/cf74.xml")
