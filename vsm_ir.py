import json
import math
import sys
import os
import xml.etree.ElementTree
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

main_dict = {}
inverted_index = {}
docs_length = {}
docs_to_denominator = {}
nltk.download('stopwords')
nltk.download('punkt')
tokenizer = RegexpTokenizer(r'\w+')
stopwords = set(stopwords.words("english"))
ps = PorterStemmer()



def extract_words_from_file(file_name):
    tree = xml.etree.ElementTree.parse(file_name)
    root = tree.getroot()
    for doc in root.findall("RECORD"):
        extract_words_from_doc(doc)


def extract_words_from_doc(doc):
    doc_id, doc_text = "", ""
    for elem in doc:
        if elem.tag == "RECORDNUM":
            doc_id = elem.text
        elif elem.tag == "TITLE" or elem.tag == "EXTRACT" or elem.tag == "ABSTRACT":
            text = str(elem.text)
            doc_text += text + " "

    text_tokens = tokenizer.tokenize(doc_text)
    filtered_text = filter_stop_words(text_tokens)  # filter stop words
    stemmed_text = words_stemming(filtered_text)  # stem text
    docs_length[doc_id] = len(stemmed_text)  # saving documents length
    doc_dict = create_doc_dict(stemmed_text)  # word to number of appearances in doc
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
            inverted_index[word] = {"df": 1,
                                    "list": {doc_id: {"f": doc_dict[word], "tf": doc_dict[word] / max_word_num}}}
        else:
            inverted_index[word]["df"] += 1
            linked_list = inverted_index[word]["list"]
            linked_list[doc_id] = {"f": doc_dict[word]}
            linked_list[doc_id]["tf"] = doc_dict[word] / max_word_num


def update_docs_length_dict():
    number_of_words = 0
    number_of_docs = len(docs_length)
    for doc_id in docs_length:
        number_of_words += docs_length[doc_id]

    docs_length["number_of_docs"] = number_of_docs
    docs_length["average_doc_length"] = number_of_words / number_of_docs


def update_docs_denominator(doc_id, grade):
    if doc_id not in docs_to_denominator:
        docs_to_denominator[doc_id] = grade ** 2
    else:
        docs_to_denominator[doc_id] += grade ** 2


def sqr_root_docs_to_denominator():
    for doc in docs_to_denominator:
        docs_to_denominator[doc] = math.sqrt(docs_to_denominator[doc])


def update_tfidf_score():
    number_of_docs = docs_length["number_of_docs"]
    for word in inverted_index:
        linked_list = inverted_index[word]["list"]
        df = inverted_index[word]["df"]
        # compute idf score
        idf = math.log2(number_of_docs / df)
        inverted_index[word]["idf"] = idf
        # compute idf score according to BM25
        numerator = number_of_docs - df + 0.5
        denominator = df + 0.5
        BM25_idf = math.log(numerator / denominator + 1)
        inverted_index[word]["BM25_idf"] = BM25_idf

        for doc_id in linked_list:
            # compute tf*idf score
            tf = linked_list[doc_id]["tf"]
            tfidf = tf * idf
            linked_list[doc_id]["tf-idf"] = tfidf
            update_docs_denominator(doc_id, tfidf)

    sqr_root_docs_to_denominator()


def create_json_file():
    with open("vsm_inverted_index.json", "w") as inverted_index_json:
        json.dump(main_dict, inverted_index_json, indent=9)


def get_max_occurrence(lst):
    dic = defaultdict(int)
    for word in lst:
        dic[word] += 1

    return max(dic.values())


def calculate_query_tf_idf_grade(question, inverted_index):
    query_set = set(question)
    grades = {}
    max_frequency = get_max_occurrence(question)
    for word in query_set:
        idf_score = inverted_index[word]["idf"] if word in inverted_index else 0
        grades[word] = (question.count(word) * idf_score) / max_frequency

    return grades


def filter_query(question):
    question = tokenizer.tokenize(question)
    question = filter_stop_words(question)
    return words_stemming(question)

def get_tfidf_query_denominator(words_to_tfidf_grade_in_query: dict):
    grades = list(words_to_tfidf_grade_in_query.values())
    sum_of_sqr = sum([grade ** 2 for grade in grades])
    return math.sqrt(sum_of_sqr)


def apply_query_with_tfidf(question, inverted_index: dict, docs_to_denominators: dict):
    relevant_docs_to_grade = {}
    question = filter_query(question)
    words_to_tfidf_grade_in_query = calculate_query_tf_idf_grade(question, inverted_index)
    query_denominator = get_tfidf_query_denominator(words_to_tfidf_grade_in_query)

    for word in set(question):
        if word not in inverted_index:
            continue
        word_relevant_docs_to_grades: dict = inverted_index[word]["list"]
        for doc in word_relevant_docs_to_grades:
            if doc not in relevant_docs_to_grade:
                relevant_docs_to_grade[doc] = word_relevant_docs_to_grades[doc]["tf-idf"] * words_to_tfidf_grade_in_query[word]
            else:
                relevant_docs_to_grade[doc] += word_relevant_docs_to_grades[doc]["tf-idf"] * words_to_tfidf_grade_in_query[word]

    for doc in relevant_docs_to_grade:
        relevant_docs_to_grade[doc] = relevant_docs_to_grade[doc] / (docs_to_denominators[doc] * query_denominator)
    return relevant_docs_to_grade


def get_bm25_grade(tf_in_doc, idf, doc_length, avg_size_of_doc):
    k1 = 2.45
    b = 0.75
    tf_numerator = tf_in_doc * (k1 + 1)
    tf_denominator = tf_in_doc + k1 * (1 - b + b*(doc_length / avg_size_of_doc))
    tf = tf_numerator / tf_denominator
    return tf * idf


def apply_query_with_bm(question, inverted_index: dict, docs: dict):
    relevant_docs_to_grade = {}
    avg_size_of_doc = docs["average_doc_length"]
    question = filter_query(question)

    for word in question:
        if word not in inverted_index:
            continue
        word_relevant_docs_to_grades: dict = inverted_index[word]["list"]
        word_idf = inverted_index[word]["BM25_idf"]

        for doc in word_relevant_docs_to_grades:
            if doc not in relevant_docs_to_grade:
                relevant_docs_to_grade[doc] = get_bm25_grade(word_relevant_docs_to_grades[doc]["f"],
                                                             word_idf,
                                                             docs[doc],
                                                             avg_size_of_doc)
            else:
                relevant_docs_to_grade[doc] += get_bm25_grade(word_relevant_docs_to_grades[doc]["f"],
                                                             word_idf,
                                                             docs[doc],
                                                             avg_size_of_doc)

    return relevant_docs_to_grade


def apply_query(ranking_method, path_to_main_dict, question):
    main_dict_as_json = open(path_to_main_dict, "r")
    main_dict = json.load(main_dict_as_json)
    inverted_index = main_dict["inverted_index"]
    docs_length = main_dict["docs_length"]
    docs_to_denominators = main_dict["docs_denominators"]

    if ranking_method == "tfidf":
        relevant_docs_to_grades = apply_query_with_tfidf(question, inverted_index, docs_to_denominators)
    elif ranking_method == "bm25":
        relevant_docs_to_grades = apply_query_with_bm(question, inverted_index, docs_length)
    else:
        raise ("invalid ranking method argument")

    return sorted(relevant_docs_to_grades.keys(), key=relevant_docs_to_grades.get, reverse=True)


def make_txt_file_of_relevant_docs(relevant_docs):
    with open("ranked_query_docs.txt", "w") as f:
        for doc in relevant_docs:
            f.write(doc + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "create_index":
        path = sys.argv[2]
        for filename in os.listdir(path):
            if filename.endswith("xml"):
                f = os.path.join(path, filename)
                extract_words_from_file(f)

        update_docs_length_dict()
        update_tfidf_score()
        main_dict["inverted_index"] = inverted_index
        main_dict["docs_length"] = docs_length
        main_dict["docs_denominators"] = docs_to_denominator
        create_json_file()

    elif sys.argv[1] == "query":
        ranking_method = sys.argv[2]
        path_to_main_dict = sys.argv[3]
        question = sys.argv[4]
        relevant_docs = apply_query(ranking_method, path_to_main_dict, question)
        relevant_docs = relevant_docs[: int(len(relevant_docs) / 6)]
        make_txt_file_of_relevant_docs(relevant_docs)

    else:
        print("invalid arguments")
