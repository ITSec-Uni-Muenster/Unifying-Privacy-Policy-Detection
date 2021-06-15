import os
import re
import string
import datetime
import statistics
import traceback
import sys
import re
from collections import Counter
from pprint import pprint
from pathlib import Path

from tinydb import TinyDB
from tinydb import where as tinydb_where
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

from chardet.universaldetector import UniversalDetector
import traceback
from joblib import Parallel, delayed, load
from tqdm import tqdm
from boilerpipe.extract import Extractor
from readabilipy import simple_json_from_html_string
from tqdm import tqdm, trange
import pandas as pd

import pycld2 as cld2
import cld3
from langdetect import detect, detect_langs, DetectorFactory, lang_detect_exception
from guess_language import guess_language
from tika import language as tika_language
import fasttext
import textacy
from textacy import preprocessing as textacy_preprocessing


import yake
import spacy
import pke
from tqdm import tqdm
from multi_rake import Rake
import textacy
import textacy.ke

from pandas.core.common import flatten
import fitz

spacy_languages = {
    "de": "de_core_news_lg",
    "el": "el_core_news_lg",
    "en": "en_core_web_lg",
    "es": "es_core_news_lg",
    "fr": "fr_core_news_lg",
    "it": "it_core_news_lg",
    "nl": "nl_core_news_lg",
    "pt": "pt_core_news_lg",
    "xx": "xx_ent_wiki_sm",
    "nb": "nb_core_news_lg",
    "lt": "lt_core_news_lg",
    "zh": "zh_core_web_lg",
    "da": "da_core_news_lg",
    "ja": "ja_core_news_lg",
    "pl": "pl_core_news_lg",
    "ro": "ro_core_news_lg",
}

data_dir = "../data/"
crawl_date = "2021-01-01" # Adapt this based on your requirements

def load_data_of_text_policies(db, language=None):
    policies_table = db.table("policies")
    list_of_policies_dicts = policies_table.all()
    print("list_of_policies_dicts: {}".format(len(list_of_policies_dicts)))

    if language:
        language_table = db.table("policies_language")
        list_of_language_dicts = language_table.search(
            tinydb_where("DeterminedLanguage") == language
        )
        print("list_of_language_dicts: {}".format(len(list_of_language_dicts)))

        list_of_language_IDs = [
            language_dict["Text_ID"] for language_dict in list_of_language_dicts
        ]
        print("list_of_language_IDs: {}".format(len(list_of_language_IDs)))

        list_of_policies_dicts = [
            policy_dict
            for policy_dict in list_of_policies_dicts
            if policy_dict["Text_ID"] in list_of_language_IDs
        ]

    list_of_texts = [
        policy_dict["Plain_Text"] for policy_dict in list_of_policies_dicts
    ]
    list_of_IDs = [policy_dict["Text_ID"] for policy_dict in list_of_policies_dicts]
    print("Number of loaded texts: {}".format(len(list_of_texts)))
    del list_of_policies_dicts
    return list_of_texts, list_of_IDs


def load_keyphrases(db, language):
    list_of_lists_of_keyphrases = []
    print("Loading keyphrases in {}".format(language))

    keyphrase_table = db.table("policies_keyphrases")
    list_of_keyphrase_dicts = keyphrase_table.all()

    language_table = db.table("policies_language")
    list_of_language_dicts = language_table.search(
        tinydb_where("DeterminedLanguage") == language
    )
    print("list_of_language_dicts: {}".format(len(list_of_language_dicts)))

    list_of_language_IDs = [
        language_dict["Text_ID"] for language_dict in list_of_language_dicts
    ]

    del list_of_language_dicts

    list_of_keyphrase_dicts = [
        keyphrases_dict
        for keyphrases_dict in list_of_keyphrase_dicts
        if keyphrases_dict["Text_ID"] in list_of_language_IDs
    ]

    list_of_TextIDs = [
        keyphrase_dict["Text_ID"] for keyphrase_dict in list_of_keyphrase_dicts
    ]

    assert sorted(list_of_TextIDs) == sorted(list_of_language_IDs)
    del list_of_language_IDs

    policies_table = db.table("policies")
    list_of_policies_dicts = policies_table.all()
    list_of_URLs = [
        policy_dict["URL"]
        for policy_dict in list_of_policies_dicts
        if policy_dict["Text_ID"] in list_of_TextIDs
    ]

    del list_of_policies_dicts

    for keyphrase_dict in list_of_keyphrase_dicts:
        list_of_keyphrases = (
            keyphrase_dict["MultiRake"]
            + keyphrase_dict["YakeOriginal"]
            + keyphrase_dict["PKE_TextRank"]
            + keyphrase_dict["PKE_SingleRank"]
            + keyphrase_dict["PKE_TopicRank"]
            + keyphrase_dict["PKE_PositionRank"]
            + keyphrase_dict["PKE_MultipartiteRank"]
            + keyphrase_dict["Textacy_sCAKE"]
        )
        list_of_lists_of_keyphrases.append(list_of_keyphrases)

    print(len(list_of_lists_of_keyphrases))
    list_of_lists_of_keyphrases = [
        list(set([keyphrase.lower() for keyphrase in list_of_keyphrases]))
        for list_of_keyphrases in list_of_lists_of_keyphrases
    ]
    return list_of_TextIDs, list_of_URLs, list_of_lists_of_keyphrases


def text_whitespace_cleaner(text):
    text = textacy_preprocessing.normalize.normalize_hyphenated_words(text)
    text = textacy_preprocessing.normalize.normalize_whitespace(text)
    text = re.sub(
        " +",
        " ",
        "".join(x if x.isprintable() or x in string.whitespace else " " for x in text),
    )
    return text


def spacy_lemmatizer_with_whitespace(texts, language):
    lemmatized_docs = []
    nlp = spacy.load(spacy_languages[language], disable=["ner"])
    for text in tqdm(texts, desc="Spacy Lemmatization"):
        nlp.max_length = len(text)
        doc = nlp(text)
        lemmatized_docs.append(
            "".join([token.lemma_ + token.whitespace_ for token in doc])
        )
    return lemmatized_docs


def text_extraction_module():

    def html_encoding_detection(page_path):
        detector = UniversalDetector()
        detector.reset()
        with open(page_path, "rb") as f:
            for line in f:
                detector.feed(line)
                if detector.done:
                    break
        detector.close()
        return detector.result["encoding"]

    def htmlfile_opener(page_path):
        encoding = html_encoding_detection(page_path)
        raw_html = None
        try:
            with open(page_path, mode="r", encoding=encoding, errors="ignore") as fin:
                raw_html = fin.read()
                fin.close()
        except:
            print(page_path)
            print(traceback.format_exc())
        return raw_html

    def text_from_html_extraction_numwordsrules(raw_html):
        text = ""
        if raw_html:
            """remove html with different settings of boilerpipe
            https://github.com/misja/python-boilerpipe
            """
            try:
                extractor = Extractor(extractor="NumWordsRulesExtractor", html=raw_html)
                text = str(extractor.getText())
            except:
                print(traceback.format_exc())
        return text

    def text_from_html_extraction_canola(raw_html):
        text = ""
        if raw_html:
            """remove html with different settings of boilerpipe
            https://github.com/misja/python-boilerpipe
            """
            try:
                extractor = Extractor(extractor="CanolaExtractor", html=raw_html)
                text = str(extractor.getText())
            except:
                print(traceback.format_exc())
        return text

    def text_from_html_extraction_readability(raw_html):
        text = ""
        result = simple_json_from_html_string(raw_html, use_readability=True)
        try:
            excerpt = result["excerpt"]
        except KeyError:
            excerpt = ""
        try:
            plain_text = result["textContent"]
        except KeyError:
            plain_text = ""
        text = excerpt + "\n\n" + plain_text
        return text

    def text_from_pdf_extractor(pdf_path):
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.getText()
        except:
            print(traceback.format_exc())

        return text

    db = TinyDB(
        os.path.join(data_dir, "database.json"),
        storage=CachingMiddleware(JSONStorage),
    )
    table = db.table("policies")

    pages = os.listdir(data_dir + "/" +  crawl_date)
    html_pages = [page for page in pages if os.path.splitext(page)[1] not in {".pdf"}]
    pdf_files = [page for page in pages if os.path.splitext(page)[1] in {".pdf"}]
    list_of_page_paths = [
        os.path.join(data_dir, "raw", page) for page in html_pages
    ]
    list_of_pdf_paths = [
        os.path.join(data_dir, "raw", page) for page in pdf_files
    ]

    list_of_raw_html_pages = Parallel(n_jobs=-1)(
        delayed(htmlfile_opener)(page_path)
        for page_path in tqdm(list_of_page_paths, desc="Encoding Detection")
    )
    assert len(list_of_raw_html_pages) == len(list_of_page_paths) == len(html_pages)
    list_of_plain_texts_numwordsrules = Parallel(n_jobs=-1)(
        delayed(text_from_html_extraction_numwordsrules)(raw_html)
        for raw_html in tqdm(list_of_raw_html_pages, desc="Text Extraction NumWordsRules")
    )
    list_of_plain_texts_canola = Parallel(n_jobs=-1)(
        delayed(text_from_html_extraction_canola)(raw_html)
        for raw_html in tqdm(list_of_raw_html_pages, desc="HTML Text Extraction Canola")
    )
    list_of_plain_texts_readability = Parallel(n_jobs=-1)(
        delayed(text_from_html_extraction_readability)(raw_html)
        for raw_html in tqdm(list_of_raw_html_pages, desc="HTML Text Extraction Readability")
    )
    assert len(list_of_plain_texts_numwordsrules) == len(list_of_plain_texts_canola) == len(list_of_plain_texts_readability) == len(html_pages) == len(list_of_raw_html_pages)

    for i, (plain_text, raw_html) in enumerate(
        zip(list_of_plain_texts_numwordsrules, list_of_raw_html_pages)
    ):
        table.upsert(
            {
                "Text_ID": "HTML_" + str(i),
                "Crawl": crawl_date,
                "URL": html_pages[i],
                "Raw": raw_html,
                "Plain_Text": plain_text,
                "Plain_Text_Canola":list_of_plain_texts_canola[i],
                "Plain_Text_Readability": list_of_plain_texts_readability[i]

            }
        )

    list_of_plain_texts = Parallel(n_jobs=-1)(
        delayed(text_from_pdf_extractor)(pdf_file_path)
        for pdf_file_path in tqdm(list_of_pdf_paths, desc="PDF Text Extraction")
    )
    assert len(list_of_plain_texts) == len(list_of_pdf_paths) == len(pdf_files)
    for i, (plain_text, pdf_filename) in enumerate(zip(list_of_plain_texts, pdf_files)):
        table.upsert(
            {
                "Text_ID": "PDF_" + str(i),
                "Crawl": crawl_date,
                "URL": pdf_filename,
                "Raw": "PDF",
                "Plain_Text": plain_text,
                "Plain_Text_Canola":"",
                "Plain_Text_Readability": ""
            }
        )
    db.close()


def language_detection_module():
    """Performs majority voting on the detected languages by the libraries"""

    def segment_multilingual_policies(vectors, text):
        """segments privacy policies by language if desired
            by using the output vectors of CLD2
        """
        list_of_segments = []
        text_as_bytes = text.encode("utf-8")
        for vector in vectors:
            start = vector[0]
            end = start + vector[1]
            segment = text_as_bytes[start:end].decode("utf-8")
            list_of_segments.append(segment)
        return list_of_segments

    def language_detection(text):

        ## prepare components ##
        DetectorFactory.seed = 0
        
        fasttext_model = fasttext.load_model("resources/lid.176.bin")

        word_re = re.compile(
            r"\w+", re.IGNORECASE | re.DOTALL | re.UNICODE | re.MULTILINE
        )

        # Just keep the words
        raw_text = textacy_preprocessing.replace.replace_urls(
            textacy_preprocessing.replace.replace_emails(text, ""), ""
        )
        raw_text = textacy_preprocessing.replace.replace_phone_numbers(raw_text, "")
        raw_text = word_re.findall(raw_text)

        if len(raw_text) > 10:
            raw_text = " ".join(raw_text)
            dict_of_detected_languages = {}
            dict_of_detection_probabilies = {}

            # 1. https://github.com/Mimino666/langdetect
            DetectorFactory.seed = 0
            try:
                dict_of_detected_languages["langdetect"] = detect(raw_text).lower()
                dict_of_detection_probabilies["langdetect_probablities"] = [
                    (item.lang, item.prob) for item in detect_langs(raw_text)
                ]
            except lang_detect_exception.LangDetectException:
                traceback.print_exc()
                dict_of_detected_languages["langdetect"] = "un"
                dict_of_detection_probabilies["langdetect_probablities"] = []

            # 2. https://github.com/aboSamoor/pycld2
            try:
                isReliable, _, details, vectors = cld2.detect(
                    raw_text, returnVectors=True
                )
                if isReliable:
                    # utf-8 bytes issue with meaningless "un"
                    dict_of_detected_languages["pycld2"] = [
                        detail[1].lower() for detail in details if detail[2] != 0
                    ]
                    dict_of_detection_probabilies["pycld2_vectors"] = list(vectors)
                else:
                    dict_of_detected_languages["pycld2"] = ["un"]
                    dict_of_detection_probabilies["pycld2_vectors"] = ()
            except:
                traceback.print_exc()
                dict_of_detected_languages["pycld2"] = ["un"]
                dict_of_detection_probabilies["pycld2_vectors"] = ()

            # 3. https://github.com/saffsd/langid.py
            try:
                from langid.langid import LanguageIdentifier, model

                langid_identifier = LanguageIdentifier.from_modelstring(
                    model, norm_probs=True
                )
                langid_tuple = langid_identifier.classify(raw_text)
                dict_of_detected_languages["langid"] = langid_tuple[0].lower()
                dict_of_detection_probabilies["langid_probability"] = langid_tuple
            except:
                traceback.print_exc()
                dict_of_detected_languages["langid"] = "un"
                dict_of_detection_probabilies["langid_probability"] = ()

            # 4. https://bitbucket.org/spirit/guess_language/
            try:
                dict_of_detected_languages["guess_language"] = guess_language(
                    raw_text
                ).lower()
            except:
                traceback.print_exc()
                dict_of_detected_languages["guess_language"] = "un"

            # 5. https://github.com/facebookresearch/fastText/tree/master/python
            # https://fasttext.cc/docs/en/language-identification.html
            try:
                dict_of_detected_languages["fasttext"] = (
                    fasttext_model.predict(raw_text)[0][0]
                    .replace("__label__", "")
                    .lower()
                )
                dict_of_detection_probabilies[
                    "fasttext_probability"
                ] = fasttext_model.predict(raw_text)[1]
            except:
                traceback.print_exc()
                dict_of_detected_languages["fasttext"] = "un"
                dict_of_detection_probabilies["fasttext_probability"] = 0

            # 6. https://github.com/chartbeat-labs/textacy/blob/master/textacy/lang_utils.py
            try:
                dict_of_detected_languages[
                    "textacy"
                ] = textacy.lang_utils.identify_lang(raw_text).lower()
            except:
                traceback.print_exc()
                dict_of_detected_languages["textacy"] = "un"

            # 7. https://github.com/bsolomon1124/pycld3
            try:
                tuple_of_detected_language = cld3.get_language(raw_text)
                isReliable = tuple_of_detected_language[2]
                if isReliable:  # is_reliable
                    dict_of_detected_languages["cld3"] = tuple_of_detected_language[
                        0
                    ].lower()
                    dict_of_detection_probabilies[
                        "cld3_probabilities"
                    ] = cld3.get_frequent_languages(raw_text, num_langs=10)
                else:
                    dict_of_detected_languages["cld3"] = "un"
                    dict_of_detection_probabilies[
                        "cld3_probabilities"
                    ] = cld3.get_frequent_languages(raw_text, num_langs=10)
            except:
                traceback.print_exc()
                dict_of_detected_languages["cld3"] = "un"
                dict_of_detection_probabilies["cld3_probabilities"] = []

            # 8. https://github.com/chrismattmann/tika-python#language-detection-interface
            try:
                dict_of_detected_languages["tika"] = tika_language.from_buffer(raw_text)
            except:
                traceback.print_exc()
                dict_of_detected_languages["tika"] = "un"

            list_of_all_detected_languages = list(
                flatten(dict_of_detected_languages.values())
            )
            list_of_all_detected_languages = [
                v if not v.startswith("zh") else "zh"
                for v in list_of_all_detected_languages
            ]
            list_of_all_detected_languages = [
                v
                if (v != "unknown" and v != "UNKNOWN" and v != "UNKNOWN_LANGUAGE")
                else "un"
                for v in list_of_all_detected_languages
            ]
            try:
                determined_language = statistics.mode(list_of_all_detected_languages)
            except statistics.StatisticsError:
                determined_language = "no-majority-achieved"

            # handling multilingual cases
            if (
                len(dict_of_detected_languages["pycld2"]) > 1
                or len(dict_of_detection_probabilies["cld3_probabilities"]) > 1
            ):
                multilingual = True
            else:
                multilingual = False

            # possibility for superflous strings as described in the paper
            if len(set(list_of_all_detected_languages))==1 and multilingual==True:
                recheck = True # check whether CanolaExtractor or Readability.js provide purer results
            else:
                recheck = False

        else:
            determined_language = "too-short-text"
            dict_of_detected_languages = {}
            dict_of_detection_probabilies = {}
            multilingual = False

        return (
            determined_language,
            dict_of_detected_languages,
            dict_of_detection_probabilies,
            multilingual,
            recheck
        )

    print("Start time: ", str(datetime.datetime.now()))
    db = TinyDB(
        os.path.join(data_dir, "database.json"),
        storage=CachingMiddleware(JSONStorage),
    )
    language_table = db.table("policies_language")

    list_of_texts, list_of_ids = load_data_of_text_policies(db, language=None)

    print("Start language detection")
    res = Parallel(n_jobs=-1)(
        delayed(language_detection)(text) for text in tqdm(list_of_texts)
    )

    print("Finished language detection")

    del list_of_texts
    list_of_determined_languages = [item[0] for item in res]  # list
    list_of_dicts_with_all_detected_languages = [item[1] for item in res]
    list_of_dicts_with_detection_probabilities = [item[2] for item in res]
    list_of_multilingual_booleans = [item[3] for item in res]
    list_of_rechecks_booleans = [item[4] for item in res]

    del res

    print(
        "Most common languages: {}".format(
            dict(Counter(list_of_determined_languages).most_common())
        )
    )

    for id, language, multilingual, recheck in zip(
        list_of_ids, list_of_determined_languages, list_of_multilingual_booleans, list_of_rechecks_booleans
    ):
        language_table.upsert(
            {
                "Text_ID": id,
                "DeterminedLanguage": language,
                "Multilingual": multilingual,
                "Recheck": recheck
            }
        )

    df = pd.DataFrame(list_of_determined_languages, columns=["DeterminedLanguage"])
    df.insert(loc=0, column="Text_ID", value=list_of_ids)
    df.insert(loc=1, column="Multilingual", value=list_of_multilingual_booleans)
    df = pd.concat(
        [
            df,
            pd.DataFrame(list_of_dicts_with_all_detected_languages),
            pd.DataFrame(list_of_dicts_with_detection_probabilities),
        ],
        axis=1,
    )


    Path("../logs/language_analysis/").mkdir(parents=True, exist_ok=True)

    df.to_json(
        "../logs/language_analysis/language_detection_probabilities_" + crawl_date + ".json",
        orient="records",
    )

    db.close()

    print("End time: ", str(datetime.datetime.now()))


def keyphrase_extraction_module():
    def multi_rake(text, language):
        r = Rake(language_code=language)
        try:
            keyphrases = r.apply(text)
            keyphrases = [keyphrase for keyphrase, score in keyphrases]
            if len(keyphrases) > 20:
                list_of_keyphrases = keyphrases[:20]
            else:
                list_of_keyphrases = keyphrases
        except:
            tqdm.write(traceback.format_exc())
            list_of_keyphrases = []
        return list_of_keyphrases

    def yake_original(text, language):
        if language == "cs":
            language = "cz"
        kwextractor = yake.KeywordExtractor(lan=language)
        try:
            keyphrases = kwextractor.extract_keywords(text)
            keyphrases = [keyphrase for keyphrase, score in keyphrases]
            if len(keyphrases) > 20:
                list_of_keyphrases = keyphrases[:20]
            else:
                list_of_keyphrases = keyphrases
        except:
            tqdm.write(traceback.format_exc())
            list_of_keyphrases = []
        return list_of_keyphrases

    def pke_textrank(text, language):
        extractor = pke.unsupervised.TextRank()
        try:
            extractor.load_document(input=text, language=language)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=20)
            list_of_keyphrases = [keyphrase for keyphrase, score in keyphrases]
        except:
            tqdm.write(traceback.format_exc())
            list_of_keyphrases = []
        return list_of_keyphrases

    def pke_singlerank(text, language):
        extractor = pke.unsupervised.SingleRank()
        try:
            extractor.load_document(input=text, language=language)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=20)
            list_of_keyphrases = [keyphrase for keyphrase, score in keyphrases]
        except:
            tqdm.write(traceback.format_exc())
            list_of_keyphrases = []
        return list_of_keyphrases

    def pke_topicrank(text, language):
        extractor = pke.unsupervised.TopicRank()
        try:
            extractor.load_document(input=text, language=language)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=20)
            list_of_keyphrases = [keyphrase for keyphrase, score in keyphrases]
        except:
            tqdm.write(traceback.format_exc())
            list_of_keyphrases = []
        return list_of_keyphrases

    def pke_positionrank(text, language):
        extractor = pke.unsupervised.PositionRank()
        try:
            extractor.load_document(input=text, language=language)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=20)
            list_of_keyphrases = [keyphrase for keyphrase, score in keyphrases]
        except:
            tqdm.write(traceback.format_exc())
            list_of_keyphrases = []
        return list_of_keyphrases

    def pke_multipartiterank(text, language):
        extractor = pke.unsupervised.MultipartiteRank()
        try:
            extractor.load_document(input=text, language=language)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=20)
            list_of_keyphrases = [keyphrase for keyphrase, score in keyphrases]
        except:
            tqdm.write(traceback.format_exc())
            list_of_keyphrases = []
        return list_of_keyphrases

    def textacy_scake(text, language):
        try:
            doc = textacy.make_spacy_doc(text, lang=language)
            keyphrases = textacy.ke.scake(doc, normalize="lemma", topn=20)
            list_of_keyphrases = [keyphrase for keyphrase, score in keyphrases]
        except:
            tqdm.write(traceback.format_exc())
            list_of_keyphrases = []
        return list_of_keyphrases

    print("Start time: ", str(datetime.datetime.now()))

    keyphrase_extractors = {
        "MultiRake": multi_rake,
        "YakeOriginal": yake_original,
        "PKE_TextRank": pke_textrank,
        "PKE_SingleRank": pke_singlerank,
        "PKE_TopicRank": pke_topicrank,
        "PKE_PositionRank": pke_positionrank,
        "PKE_MultipartiteRank": pke_multipartiterank,
        "Textacy_sCAKE": textacy_scake,
    }

    groups_of_algorithms = {
        "misc": ["MultiRake", "YakeOriginal"],
        "pke": [
            "PKE_TextRank",
            "PKE_SingleRank",
            "PKE_TopicRank",
            "PKE_PositionRank",
            "PKE_MultipartiteRank",
        ],
        "textacy": ["Textacy_sCAKE"],
    }

    list_of_languages = ["en", "de"]

    db = TinyDB(
        os.path.join(data_dir, "database.json"),
        storage=CachingMiddleware(JSONStorage),
    )

    lemmatized_table = db.table("policies_lemmatized")
    keyphrase_table = db.table("policies_keyphrases")

    for language in list_of_languages:
        list_of_keyphrase_dicts = []

        list_of_texts, list_of_IDs = load_data_of_text_policies(db, language=language)

        list_of_texts = Parallel(n_jobs=-1)(
            delayed(text_whitespace_cleaner)(text)
            for text in tqdm(list_of_texts, desc="Cleaning texts")
        )
        list_of_lemmatized_texts = spacy_lemmatizer_with_whitespace(
            list_of_texts, language
        )
        for ID, lemmatized_text in zip(tqdm(list_of_IDs), list_of_lemmatized_texts):
            lemmatized_table.upsert(
                {"Text_ID": ID, "Language": language, "Lemmatized_Text": lemmatized_text}
            )

        for ID in tqdm(list_of_IDs, desc="List of dicts"):
            list_of_keyphrase_dicts.append({"Text_ID": ID})
        print("len(list_of_keyphrase_dicts):", len(list_of_keyphrase_dicts))

        # Depending whether the library does lemmatization or not by itself, the appropriate list is passed to the function
        for name, extractor in keyphrase_extractors.items():
            if name in groups_of_algorithms["textacy"]:
                list_of_lists_of_keywords = Parallel(n_jobs=-1)(
                    delayed(extractor)(text, language)
                    for text in tqdm(list_of_texts, desc=name)
                )
            else:
                list_of_lists_of_keywords = Parallel(n_jobs=-1)(
                    delayed(extractor)(text, language)
                    for text in tqdm(list_of_lemmatized_texts, desc=name)
                )

            for i, ID in enumerate(tqdm(list_of_IDs)):
                if list_of_keyphrase_dicts[i]["Text_ID"] == ID:
                    list_of_keyphrase_dicts[i] = {
                        **list_of_keyphrase_dicts[i],
                        **{name: list_of_lists_of_keywords[i]},
                    }
        assert len(list_of_IDs) == len(list_of_keyphrase_dicts)
        for ID, keyphrase_dicts in zip(tqdm(list_of_IDs), list_of_keyphrase_dicts):
            keyphrase_table.upsert(keyphrase_dicts, tinydb_where("Text_ID") == ID)

    db.close()

    print("End time: ", str(datetime.datetime.now()))


def policy_detection_module():
    def keyphrase_analyzer(list_of_list_of_keyphrases):
        all_keywords = []
        list_of_dict_keyphrases = []
        number_of_policies = str(len(list_of_list_of_keyphrases))
        for list_of_keyphrases in list_of_list_of_keyphrases:
            all_keywords += list_of_keyphrases
            dict_of_keyphrases = dict(
                Counter(list_of_keyphrases)
            )
            list_of_dict_keyphrases.append(dict_of_keyphrases)
        print(number_of_policies + " policies:", Counter(all_keywords).most_common(50))
        print("#unique keywords:", len(set(all_keywords)))

        return list_of_dict_keyphrases

    def label_comparison(
        language, list_of_dict_keyphrases, list_of_TextIDs, list_of_URLs
    ):
        print("Loading vectorizer and classifier")
        vectorizer = load(
            "resources/trained_vectorizer_" + language + ".pkl"
        )
        clf = load(
            "resources/VotingClassifier_soft_" + language + ".pkl"
        )

        print("Vectorizer transformation")
        X_unlabeled = vectorizer.transform(list_of_dict_keyphrases)

        print("Shape of unlabeled texts: {}".format(X_unlabeled.shape))
        print("prediction")
        y_pred = clf.predict(X_unlabeled)
        y_pred_proba = clf.predict_proba(X_unlabeled)

        print(Counter(y_pred))
        df = pd.DataFrame()
        df["TextID"] = list_of_TextIDs
        df["URL"] = list_of_URLs
        df["Language"] = language
        df["PredictedLabels"] = y_pred
        df_proba = pd.DataFrame(
            y_pred_proba, columns=["Probability_0", "Probability_1"]
        )
        df = pd.concat([df, df_proba], axis=1)
        df.to_csv("../results/classification/classification_" + language + ".csv")

    print("Start time: ", str(datetime.datetime.now()))

    db = TinyDB(
        os.path.join(data_dir, "database.json"),
        storage=CachingMiddleware(JSONStorage),
    )

    list_of_languages = ["en", "de"]

    for language in tqdm(list_of_languages, unit="language"):
        (
            list_of_TextIDs,
            list_of_URLs,
            list_of_lists_of_keyphrases,
        ) = load_keyphrases(db, language)
        list_of_dict_keyphrases = keyphrase_analyzer(list_of_lists_of_keyphrases)
        label_comparison(
            language, list_of_dict_keyphrases, list_of_TextIDs, list_of_URLs
        )

    db.close()
    print("End time: ", str(datetime.datetime.now()))


if __name__ == "__main__":
    text_extraction_module()
    language_detection_module()
    keyphrase_extraction_module()
    policy_detection_module()