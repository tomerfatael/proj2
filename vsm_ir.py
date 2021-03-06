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
    docs_length[doc_id] = len(text_tokens)  # saving documents length TODO check the right place
    filtered_text = filter_stop_words(text_tokens)  # filter stop words
    stemmed_text = words_stemming(filtered_text)  # stem text
    doc_dict = create_doc_dict(stemmed_text)  # word to number of appearances in doc
    update_inverted_index(doc_id, doc_dict)


def filter_stop_words(text_tokens):
    return [w for w in text_tokens if not w.lower() in stopwords]


def words_stemming(filtered_text):
    return [ps.stem(w) for w in filtered_text]  # TODO check about duplications


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
            linked_list[doc_id]["tf-idf"] = tf * idf


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
        grades[word] = (question.count(word) * inverted_index[word]["idf"]) / max_frequency

    return grades


def filter_query(question):
    question = tokenizer.tokenize(question)
    question = filter_stop_words(question)
    return words_stemming(question)
    # return words_stemming(filter_stop_words(tokenizer.tokenize(question)))  TODO- single row, replace after debug.


def get_tfidf_query_denominator(words_to_tfidf_grade_in_query: dict):
    grades = list(words_to_tfidf_grade_in_query.values())
    sum_of_sqr = sum([grade ** 2 for grade in grades])
    return math.sqrt(sum_of_sqr)


def apply_query_with_tfidf(question, inverted_index):
    relevant_docs_to_grade = {}
    question = filter_query(question)
    words_to_tfidf_grade_in_query = calculate_query_tf_idf_grade(question, inverted_index) # ?????????? ???? ???????? ???????? ??????
    query_denominator = get_tfidf_query_denominator(words_to_tfidf_grade_in_query)
    docs_denominator = inverted_index["docs_denominator"] # TODO fix

    for word in question:
        word_relevant_docs_to_grades: dict = inverted_index[word]["list"]
        for doc in word_relevant_docs_to_grades.keys(): # computing only the numerator
            if doc not in relevant_docs_to_grade:
                relevant_docs_to_grade[doc] = word_relevant_docs_to_grades[doc] * words_to_tfidf_grade_in_query[word]
            else:
                relevant_docs_to_grade[doc] += word_relevant_docs_to_grades[doc] * words_to_tfidf_grade_in_query[word]

    for doc in relevant_docs_to_grade.keys():
        relevant_docs_to_grade[doc] = relevant_docs_to_grade[doc] / (docs_denominator[doc] * query_denominator)
    return relevant_docs_to_grade


def get_bm25_grade(tf_in_doc, idf, doc_length, avg_size_of_doc):
    k1 = 1.2
    b = 0.75
    tf = (tf_in_doc * (k1 + 1)) / (tf_in_doc + (k1 * (1 - b + b * (doc_length / avg_size_of_doc))))
    return tf * idf


def apply_query_with_bm(question, inverted_index):
    relevent_docs_to_grade = {}
    avg_size_of_doc = inverted_index["docs"]["avg"]
    question = filter_query(question)

    for word in question:
        word_relevant_docs_to_grades: dict = inverted_index[word][
            "list"]  # TODO check this is the right way to use the dict
        for doc in word_relevant_docs_to_grades.keys():  # computing only the numerator
            if doc not in relevent_docs_to_grade:
                relevent_docs_to_grade[doc] = get_bm25_grade(inverted_index[word]["list"][doc],
                                                             inverted_index[word]["list"]["idfbm25"],
                                                             avg_size_of_doc)  # TODO ?????????? ???? ?????????? ???? ???????????? ?????? ?????????? ?????? ???????????? ?????????? ????????????
            else:
                relevent_docs_to_grade[doc] += get_bm25_grade(inverted_index[word]["list"][doc],
                                                              inverted_index[word]["list"]["idfbm25"], avg_size_of_doc)

    return relevent_docs_to_grade


def apply_query(ranking_method, path_to_main_dict, question):
    main_dict_as_json = open(path_to_main_dict, "r")
    main_dict = json.load(main_dict_as_json) # inverted index should be the map as we built it, see if there are changes that needs to be done.
    inverted_index = main_dict["inverted_index"]
    docs_length = main_dict["docs_length"]

    if ranking_method == "tfidf":
        relevant_docs_to_grades = apply_query_with_tfidf(question, inverted_index)
    elif ranking_method == "bm25":
        relevant_docs_to_grades = apply_query_with_bm(question, inverted_index)
    else:
        raise ("invalid ranking method argument")

    return sorted(relevant_docs_to_grades.keys(), key=relevant_docs_to_grades.get, reverse=True)  # TODO test logic


def make_txt_file_of_relevant_docs(relevant_docs):
    pass # need to be done


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
        create_json_file()

    elif sys.argv[1] == "query":
        ranking_method = sys.argv[2]
        path_to_main_dict = sys.argv[3]
        question = sys.argv[4]
        relevant_docs = apply_query(ranking_method, path_to_main_dict, question)
        make_txt_file_of_relevant_docs(relevant_docs)

    else:
        print("invalid arguments")
