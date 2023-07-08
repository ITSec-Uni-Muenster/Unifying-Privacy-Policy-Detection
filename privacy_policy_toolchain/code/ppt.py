import os
import re
import time
import string
import datetime
import statistics
import traceback
import sys
import re
from collections import Counter
from pprint import pprint
from pathlib import Path
from urllib import request

from tinydb import TinyDB, Query
from tinydb import where as tinydb_where
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

from chardet.universaldetector import UniversalDetector
from joblib import Parallel, delayed, load
import psutil
import stopit
from tqdm import tqdm
from boilerpipe.extract import Extractor
from readabilipy import simple_json_from_html_string
from bs4 import BeautifulSoup
from html_sanitizer import Sanitizer
from markdownify import markdownify as md
from tqdm import tqdm, trange
import pandas as pd

from publicsuffix2 import PublicSuffixList
from publicsuffix2 import get_public_suffix
import tldextract
from tld import get_tld, get_fld
from urllib.parse import unquote, urlparse

import pycld2 as cld2
import cld3
from langdetect import detect, detect_langs, DetectorFactory, lang_detect_exception
from guess_language import guess_language
import fasttext
import textacy
from textacy import preprocessing as textacy_preprocessing
import ftfy

import ndjson
import ujson
import json

import yake
import spacy
import pke
from tqdm import tqdm
from multi_rake import Rake

from difflib import SequenceMatcher

import hashlib
import simhash

from pandas.core.common import flatten
import fitz

sanitizer = Sanitizer()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

dict_of_umlaute_errors = {'Ã¼':'ü',
                            'Ã¤':'ä',
                            'Ã¶':'ö',
                            'Ã–':'Ö',
                            'ÃŸ':'ß',
                            'Ã ':'à',
                            'Ã¡':'á',
                            'Ã¢':'â',
                            'Ã£':'ã',
                            'Ã¹':'ù',
                            'Ãº':'ú',
                            'Ã»':'û',
                            'Ã™':'Ù',
                            'Ãš':'Ú',
                            'Ã›':'Û',
                            'Ãœ':'Ü',
                            'Ã²':'ò',
                            'Ã³':'ó',
                            'Ã´':'ô',
                            'Ã¨':'è',
                            'Ã©':'é',
                            'Ãª':'ê',
                            'Ã«':'ë',
                            'Ã€':'À',
                            'Ã':'Á',
                            'Ã‚':'Â',
                            'Ãƒ':'Ã',
                            'Ã„':'Ä',
                            'Ã…':'Å',
                            'Ã‡':'Ç',
                            'Ãˆ':'È',
                            'Ã‰':'É',
                            'ÃŠ':'Ê',
                            'Ã‹':'Ë',
                            'ÃŒ':'Ì',
                            'Ã':'Í',
                            'ÃŽ':'Î',
                            'Ã':'Ï',
                            'Ã‘':'Ñ',
                            'Ã’':'Ò',
                            'Ã“':'Ó',
                            'Ã”':'Ô',
                            'Ã•':'Õ',
                            'Ã˜':'Ø',
                            'Ã¥':'å',
                            'Ã¦':'æ',
                            'Ã§':'ç',
                            'Ã¬':'ì',
                            'Ã­':'í',
                            'Ã®':'î',
                            'Ã¯':'ï',
                            'Ã°':'ð',
                            'Ã±':'ñ',
                            'Ãµ':'õ',
                            'Ã¸':'ø',
                            'Ã½':'ý',
                            'Ã¿':'ÿ',
                            'â‚¬':'€'}

dict_of_umlaute_errors = {**dict_of_umlaute_errors,
                          **{key.lower(): value for key, value in dict_of_umlaute_errors.items()}}

if len(sys.argv) != 2:
    print("Please give the folder containing the raw files as input. For example: python ppt.py /home/me/privacypolicies", flush=True)
    sys.exit()
else:
    data_dir = sys.argv[1]
    crawl = data_dir.split("/")[-1].lstrip("datadir_")
    print("Working on Crawl:", crawl, flush=True)


def load_data_of_text_policies(db, language=None):
    policies_table = db.table("policies")
    list_of_policies_dicts = policies_table.all()
    print("list_of_policies_dicts: {}".format(len(list_of_policies_dicts)), flush=True)

    if language:
        language_table = db.table("policies_language")
        list_of_language_dicts = language_table.search(
            tinydb_where("DeterminedLanguage") == language
        )
        print("list_of_language_dicts: {}".format(len(list_of_language_dicts)), flush=True)

        list_of_language_IDs = [
            language_dict["TextID"] for language_dict in list_of_language_dicts
        ]
        print("list_of_languageIDs: {}".format(len(list_of_language_IDs)), flush=True)

        list_of_policies_dicts = [
            policy_dict
            for policy_dict in list_of_policies_dicts
            if policy_dict["TextID"] in list_of_language_IDs
        ]

    list_of_texts = [
        policy_dict["Text"] for policy_dict in list_of_policies_dicts
    ]
    list_of_IDs = [policy_dict["TextID"] for policy_dict in list_of_policies_dicts]
    print("Number of loaded texts: {}".format(len(list_of_texts)), flush=True)
    del list_of_policies_dicts
    return list_of_texts, list_of_IDs


def text_cleaner(text):

    def fix_utf8_iso8859_errors(text):
        # source: https://sebastianviereck.de/mysql-php-umlaute-sonderzeichen-utf8-iso/
        for error, replacement in dict_of_umlaute_errors.items():
            text = text.replace(error, replacement)
        return text

    text = textacy_preprocessing.normalize.bullet_points(text)
    text = textacy_preprocessing.normalize.unicode(text)
    text = ftfy.fix_text(text)
    text = fix_utf8_iso8859_errors(text)
    text = textacy_preprocessing.normalize.hyphenated_words(text)
    text = textacy_preprocessing.normalize.whitespace(text)
    text = textacy_preprocessing.replace.emails(text, "REPLACEDEMAIL")
    text = textacy_preprocessing.replace.urls(text, "REPLACEDURL")
    text = textacy_preprocessing.replace.phone_numbers(text, "REPLACEDPHONENUMBER")
    text = re.sub(
        " +",
        " ",
        "".join(x if x.isprintable() or x in string.whitespace else " " for x in text),
    )
    text = text.replace("\n", "\n\n")
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

def domain_cleaner(domain):
    domain = domain.lower()
    if domain.startswith("http://"):
        domain = domain.replace("http://", "", 1)
    elif domain.startswith("https://"):
        domain = domain.replace("https://", "", 1)
    else:
        domain = domain
    return domain

def get_policy_domain(url):
    url = url.lower() # lowercase everything
    url = "".join(url.splitlines()) # remove line breaks
    if url.startswith("http_"):
        url = url.replace("http_", "", 1)
    elif url.startswith("https_"):
        url = url.replace("https_", "", 1)
    if len(url.split("_")[0]) > 1:
        url = url.split("_")[0]
    if url.endswith("443"):
        url = url.rstrip("443")
    elif url.endswith("40018"):
        url = url.rstrip("40018")
    elif url.endswith("8090"):
        url = url.rstrip("8090")
    elif url.endswith("80"):
        url = url.rstrip("80")
    elif url.endswith("809"):
        url = url.rstrip("809")
    try:
        domain = get_fld(url, fail_silently=False, fix_protocol=True)
    except:
        domain = urlparse(url).netloc
    return domain


def stripprotocol(uri):
    noprotocolluri = ""
    if uri.find("https") == 0:
        noprotocolluri = uri[5:]
    else:
        noprotocolluri = uri[4:]
    return noprotocolluri


def filenamesplitter(filename):
    identifier = filename.split("_")
    host = identifier[0]
    uri = identifier[1]
    crawl = identifier[len(identifier) - 1]
    # if the uri contained an underscore this reconstructs the full uri
    if len(identifier) > 3:
        uri = ""
        for part in identifier:
            if identifier.index(part) not in [0, len(identifier) - 1]:
                uri += part + "_"
        uri = uri[:-1]
        uri = stripprotocol(uri)
    return host, uri, crawl


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
        with open(page_path, mode="r", encoding=encoding, errors="ignore") as fin:
            raw_html = fin.read()
            fin.close()
        return raw_html

    def text_from_html_extraction_canola(raw_html):
        """Remove HTML/XML using Conola settings of Boilerpipe
        https://github.com/misja/python-boilerpipe
        """
        text = ""
        if raw_html:
            try:
                extractor = Extractor(extractor="CanolaExtractor", html=raw_html)
                text = str(extractor.getText())
            except:
                traceback.print_exc()
        return text

    def text_from_html_extraction_keepeverything(raw_html):
        """Remove HTML/XML using Conola settings of Boilerpipe
        https://github.com/misja/python-boilerpipe
        """

        text = ""
        if raw_html:
            try:
                extractor = Extractor(extractor="KeepEverythingExtractor", html=raw_html)
                text = str(extractor.getText())
            except:
                traceback.print_exc()
        return text

    def text_from_html_extraction_readability(raw_html):
        """
        remove HTML/XML with
        https://github.com/alan-turing-institute/ReadabiliPy
        """
        text = ""
        timeout = 10
        try:
            # throws memory erros for > 6.3MB files dispite updating node.js and Readability.js (2023.02.07)
            # Alternativly "while psutil.virtual_memory.percent < 50" but does not determine the amuont of memory this function/process is using. 
            with stopit.ThreadingTimeout(timeout) as context_manager:
                # https://theautomatic.net/2021/11/27/how-to-stop-long-running-code-in-python/
                result = simple_json_from_html_string(raw_html, use_readability=True)
                title = result["title"]
                if title is None:
                    title = ""
                plain_text = result["plain_text"][-1]["text"]
                if plain_text is None:
                    plain_text = ""
                text = title + "\n\n" + plain_text
            if context_manager.state == context_manager.TIMED_OUT:
                text = ""
            elif context_manager.state == context_manager.EXECUTED:
                pass
        except:
            text = ""
        return text

    def text_from_html_extraction_numwordsrules(raw_html):
        """Remove HTML/XML using NumWordsRules setting of Boilerpipe
        https://github.com/misja/python-boilerpipe
        """
        text = ""
        if raw_html:
            try:
                extractor = Extractor(extractor="NumWordsRulesExtractor", html=raw_html)
                text = str(extractor.getText())
            except:
                traceback.print_exc()
        return text

    def markdown_from_html_extraction_markdownify(raw_html):
        """Convert HTML/XML to Markdown format using
        https://github.com/matthewwithanm/python-markdownify
        """
        text = ""
        try:
            unwanted_tags = ["nav", "header", "footer"]
            soup = BeautifulSoup(raw_html, "lxml")
            _ = [tag.decompose() for tag in soup(unwanted_tags)]
            text = md(str(soup))
            # body = soup.find("body")
            # text = md(raw_html)

        except:
            traceback.print_exc()
            sys.exit()
        return text

    def text_from_pdf_extractor(pdf_path):
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
        except:
            traceback.print_exc()

        return text

    def makeSimhash(text):
        #https://github.com/seomoz/simhash-py/issues/47
        import ctypes
        list_of_tokens = re.split('\s+', re.sub(r'[^\w\s]', '', text.lower()))
        # A generator for ' '-joined strings of consecutive tokens
        shingles = (' '.join(tokens) for tokens in simhash.shingle(list_of_tokens, 4))
        # They need to be unsigned 64-bit ints
        return simhash.compute([ctypes.c_ulong(hash(shingle)).value for shingle in shingles])

    def makeSHA1hash(text):
        hashvalue = hashlib.sha1(text.encode()).hexdigest()
        return hashvalue

    def process_policies(item, data_dir, california_policy, page):
        temp_dict = None
        landing_text = ""
        plain_text = ""
        plain_text_readability = ""
        plain_text_canola = ""
        markdown_text = ""
        try:
            landing_page_path = item["landing_page"]
            landing_page_path = os.path.join(data_dir, "landing_pages", landing_page_path.split("/")[-1])
            raw_landing_html = htmlfile_opener(landing_page_path) # Encoding Detection
            landing_html = sanitizer.sanitize(raw_landing_html)
            # Debug: print("sanitized landing page", flush=True)
            landing_text = text_from_html_extraction_keepeverything(landing_html)
            # Debug: print("keepeverything landing page", flush=True)
            if california_policy is False:
                page_path = os.path.join(data_dir, "privacy_policies", page.split("/")[-1])
            elif california_policy is True:
                page_path = os.path.join(data_dir, "california_privacy_pages", page.split("/")[-1])
            if os.path.splitext(page)[1] not in {".pdf"}:
                raw_html = htmlfile_opener(page_path) # Encoding Detection
                # Debug: print("opened privacy policy", flush=True)
                raw_html = sanitizer.sanitize(raw_html)
                # Debug: print("sanitized privacy policy", flush=True)
                plain_text = text_from_html_extraction_numwordsrules(raw_html)
                # Debug: print("numworldsrules privacy policy", flush=True)
                plain_text_readability = text_from_html_extraction_readability(raw_html) 
                # Debug: print("readability privacy policy", flush=True)
                plain_text_canola = text_from_html_extraction_canola(raw_html)
                # Debug: print("canola privacy policy", flush=True)
                markdown_text = markdown_from_html_extraction_markdownify(raw_html)
                # Debug: print("markdown privacy policy", flush=True)
            elif os.path.splitext(page)[1] in {".pdf"}:
                plain_text = text_from_pdf_extractor(page_path)
            temp_dict = {
                "TextID": str(i) + "_" + crawl,
                "CrawlID": item["crawl_id"],
                "Domain_origin": domain_cleaner(item["domain"]),
                "Landing_URL": item["landing_url"],
                "URL": url,
                "Policy_domain": get_fld(unquote(url.lstrip("%3A%2F%2F")), fail_silently=True, fix_protocol=True),
                "Crawl": crawl,
                "Landing_Text": landing_text,
                "Text": plain_text,
                "Text_Canola":plain_text_canola,
                "Text_Readability": plain_text_readability,
                "Text_Markdown": markdown_text,
                "SHA1": makeSHA1hash(plain_text),
                "Simhash": makeSimhash(plain_text),
                "Type":"californiapolicy" if california_policy is True else "privacypolicy"
            }
        except:
            print(page_path, flush=True)
            traceback.print_exc()
            temp_dict = None
        return temp_dict


    storage = CachingMiddleware(JSONStorage)
    storage.WRITE_CACHE_SIZE = 25
    db = TinyDB(
        os.path.join(data_dir, "CPRA_policies_database_" + crawl + ".json"),
        storage=storage
    )
    table_policies = db.table("policies")

    # function entry point
    # read existing data
    pages = os.listdir(data_dir)
    if len(pages) == 0:
        print("The folder you specified (" + data_dir + ") does not contain any files", flush=True)
        sys.exit()

    with open(data_dir + "/crawl-data.ndjson", encoding="utf-8", mode="r") as f:
        i = 1
        reader = ndjson.reader(f)
        for item in tqdm(reader, desc="Domain item"):
            # try:
                # item = json.loads(line.strip())
                # 'domain'
                # 'crawl_id'
                # 'landing_url'
                # 'landing_page'
                # 'privacy_policy_url'
                # 'privacy_policy_file'
                # 'california_url'
                # 'california_file'
            # except json.decoder.JSONDecodeError as e:
            #     traceback.print_exc()
            #     print(line[e.pos-5:e.pos+5])
            #     print()
            
            try:
                assert len(item["privacy_policy_file"]) == len(item["privacy_policy_url"])
            except AssertionError:
                print(item, flush=True)
                continue

            try:
                assert len(item["california_url"]) == len(item["california_file"])
            except AssertionError:
                print(item, flush=True)
                continue
            if len(item["privacy_policy_file"]) > 0: # if a privacy policy was found during the crawl
                for url, page in zip(item["privacy_policy_url"], item["privacy_policy_file"]): # real url and file on hard drive
                    try:
                        policy = Query()
                        result = table_policies.search((policy.Domain_origin == domain_cleaner(item["domain"])) & (policy.URL == url) & (policy.Landing_URL == item["landing_url"]) & (policy.Type == "privacypolicy"))
                        if len(result) == 0:
                            # Debug: print(f'Processing privacy policy {page} of {item["domain"]}', flush=True)
                            temp_dict = process_policies(item, data_dir, False, page)
                            if temp_dict is not None:
                                # Debug: print(f'Inserting privacy policy {page} of {item["domain"]}', flush=True)
                                table_policies.upsert(temp_dict, tinydb_where("TextID") == str(i) + "_" + crawl)
                                i = i+1
                            else:
                                continue
                        elif len(result) == 1:
                            print("Already exists in the database:", url, flush=True)
                            i = i+1
                            continue
                        elif len(result) > 1:
                            print("Too many results for", domain_cleaner(item["domain"]), flush=True)
                            continue
                    except:
                        traceback.print_exc()
                        continue
            if len(item["california_file"]) > 0: # if a california policy was found during the crawl
                for url, page in zip(item["california_url"], item["california_file"]): # real url and url on hard drive
                    try:
                        policy = Query()
                        result = table_policies.search((policy.Domain_origin == domain_cleaner(item["domain"])) & (policy.URL == url) & (policy.Landing_URL == item["landing_url"]) & (policy.Type == "californiapolicy"))
                        if len(result) == 0:
                            # Debug: print(f'Processing california policy {page} of {item["domain"]}', flush=True)
                            temp_dict = process_policies(item, data_dir, True, page)
                            if temp_dict is not None:
                                # Debug: print(f'Processing california policy {page} of {item["domain"]}', flush=True)
                                table_policies.upsert(temp_dict, tinydb_where("TextID") == str(i) + "_" + crawl)
                                i = i+1
                            else:
                                continue
                        elif len(result) == 1:
                            print("Already exists in the database:", url, flush=True)
                            i = i+1
                            continue
                        elif len(result) > 1:
                            print("Too many results for", domain_cleaner(item["domain"]), flush=True)
                    except:
                        traceback.print_exc()
                        continue
    db.close()
    print("End time: ", str(datetime.datetime.now()), flush=True)

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

        fasttext_model = fasttext.load_model("./code/resources/lid.176.bin")

        word_re = re.compile(
            r"\w+", re.IGNORECASE | re.DOTALL | re.UNICODE | re.MULTILINE
        )

        # Just keep the words
        raw_text = textacy_preprocessing.replace.urls(
            textacy_preprocessing.replace.emails(text, ""), ""
        )
        raw_text = textacy_preprocessing.replace.phone_numbers(raw_text, "")
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
                ] = textacy.identify_lang(raw_text).lower()
            except:
                #traceback.print_exc()
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

            list_of_all_detected_languages = list(
                flatten(dict_of_detected_languages.values())
            )
            list_of_all_detected_languages = [
                v if not v.startswith("zh") else "zh"
                for v in list_of_all_detected_languages
            ]
            list_of_all_detected_languages = [
                v
                if (v not in ("unknown", "UNKNOWN", "UNKNOWN_LANGUAGE"))
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
            if len(set(list_of_all_detected_languages))==1 and multilingual is True:
                recheck = True # Mark to check whether CanolaExtractor or Readability.js could provide purer plain text
            else:
                recheck = False

        else:
            determined_language = "too-short-text"
            dict_of_detected_languages = {}
            dict_of_detection_probabilies = {}
            multilingual = False
            recheck = False

        return (
            determined_language,
            dict_of_detected_languages,
            dict_of_detection_probabilies,
            multilingual,
            recheck
        )

    print("Start time: ", str(datetime.datetime.now()), flush=True)
    storage = CachingMiddleware(JSONStorage)
    storage.WRITE_CACHE_SIZE = 25
    db = TinyDB(
        os.path.join(data_dir, "CPRA_policies_database_" + crawl + ".json"),
        storage=storage
    )
    language_table = db.table("policies_language")

    list_of_texts, list_of_ids = load_data_of_text_policies(db, language=None)

    if not os.path.exists("./code/resources/lid.176.bin"):
        Path("./code/resources/").mkdir(parents=True, exist_ok=True)

        print("Downloading language model of FastText ...", flush=True)
        request.urlretrieve("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", "code/resources/lid.176.bin")


    print("Start language detection", flush=True)
    res = Parallel(n_jobs=-1)(
        delayed(language_detection)(text) for text in tqdm(list_of_texts)
    )

    print("Finished language detection", flush=True)

    del list_of_texts
    list_of_determined_languages = [item[0] for item in res]  # list
    list_of_dicts_with_all_detected_languages = [item[1] for item in res]
    list_of_dicts_with_detection_probabilities = [item[2] for item in res]
    list_of_multilingual_booleans = [item[3] for item in res]
    list_of_rechecks_booleans = [item[4] for item in res]

    del res

    print(
        "Most common languages: {}".format(
            dict(Counter(list_of_determined_languages).most_common(), flush=True)
        )
    )

    for id, language, multilingual, recheck in zip(
        list_of_ids, list_of_determined_languages, list_of_multilingual_booleans, list_of_rechecks_booleans
    ):
        language_table.upsert(
            {
                "TextID": id,
                "DeterminedLanguage": language,
                "Multilingual": multilingual,
                "Recheck": recheck
            }, tinydb_where("TextID") == id
        )

    df = pd.DataFrame(list_of_determined_languages, columns=["DeterminedLanguage"])
    df.insert(loc=0, column="TextID", value=list_of_ids)
    df.insert(loc=1, column="Multilingual", value=list_of_multilingual_booleans)
    df = pd.concat(
        [
            df,
            pd.DataFrame(list_of_dicts_with_all_detected_languages),
            pd.DataFrame(list_of_dicts_with_detection_probabilities),
        ],
        axis=1,
    )


    Path("./logs/language_analysis/").mkdir(parents=True, exist_ok=True)

    df.to_json(
        "./logs/language_analysis/language_detection_probabilities_" + crawl + ".json",
        orient="records",
    )

    db.close()

    print("End time: ", str(datetime.datetime.now()), flush=True)


def keyphrase_extraction_module():
    def multi_rake(text, language):
        # https://pypi.org/project/multi-rake/
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
        # https://pypi.org/project/yake/
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
        # https://github.com/boudinfl/pke
        extractor = pke.unsupervised.TextRank()
        try:
            extractor.load_document(input=text, language=language, normalization="none")
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
            extractor.load_document(input=text, language=language, normalization="none")
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
            extractor.load_document(input=text, language=language, normalization="none")
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
            extractor.load_document(input=text, language=language, normalization="none")
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
            extractor.load_document(input=text, language=language, normalization="none")
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
            doc = textacy.make_spacy_doc(text, lang=spacy_languages[language])
            keyphrases = textacy.extract.keyterms.scake(doc, normalize="lemma", topn=20)
            list_of_keyphrases = [keyphrase for keyphrase, score in keyphrases]
        except:
            tqdm.write(traceback.format_exc())
            list_of_keyphrases = []
        return list_of_keyphrases

    print("Start time: ", str(datetime.datetime.now()), flush=True)

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

    list_of_languages = ["de", "en"]

    storage = CachingMiddleware(JSONStorage)
    storage.WRITE_CACHE_SIZE = 25
    db = TinyDB(
        os.path.join(data_dir, "CPRA_policies_database_" + crawl + ".json"),
        storage=storage
    )

    lemmatized_table = db.table("policies_lemmatized")
    keyphrase_table = db.table("policies_keyphrases")

    for language in list_of_languages:

        list_of_texts, list_of_IDs = load_data_of_text_policies(db, language=language)

        if len(list_of_IDs) == 0 or len(list_of_texts) == 0:
            print("Texts were not loaded properly!", flush=True)
            sys.exit(0)

        list_of_texts = Parallel(n_jobs=-1)(
            delayed(text_cleaner)(text)
            for text in tqdm(list_of_texts, desc="Cleaning texts")
        )
        list_of_lemmatized_texts = spacy_lemmatizer_with_whitespace(
            list_of_texts, language
        )
        for ID, lemmatized_text in zip(tqdm(list_of_IDs, desc="Save lemmatized text"), list_of_lemmatized_texts):
            lemmatized_table.upsert(
                {"TextID": ID, "Language": language, "Lemmatized_Text": lemmatized_text},
                tinydb_where("TextID") == ID
            )

        ### MULTIPROCESSING VERSION ###
        list_of_keyphrase_dicts = []
        for ID in tqdm(list_of_IDs, desc="List of keyphrase dicts"):
            list_of_keyphrase_dicts.append({"TextID": ID, "Keyphrases":set()})
        print("len(list_of_keyphrase_dicts):", len(list_of_keyphrase_dicts), flush=True)

        # Depending on whether the library does lemmatization itself or not, the appropriate list is passed to the function
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

            for i, ID in enumerate(list_of_IDs):
                if list_of_keyphrase_dicts[i]["TextID"] == ID:
                    list_of_keyphrase_dicts[i]["Keyphrases"].update(set(list_of_lists_of_keywords[i]))
                    # list_of_keyphrase_dicts[i] = {
                    #     **list_of_keyphrase_dicts[i],
                    #     **{name: list_of_lists_of_keywords[i]},
                    # }

        # tinydb does not like sets as they are not serialiseable
        for keyphrase_dict in list_of_keyphrase_dicts:
            keyphrase_dict["Keyphrases"] = list(keyphrase_dict["Keyphrases"])

        print("Saving extracted keyphrases", flush=True)
        assert len(list_of_IDs) == len(list_of_keyphrase_dicts)
        for ID, keyphrase_dict in zip(tqdm(list_of_IDs), list_of_keyphrase_dicts):
            keyphrase_table.upsert(keyphrase_dict, tinydb_where("TextID") == ID)

        ### SINGLE PROCESSING VERSION IF MULTIPROCESSING DOES NOT WORK ###
        # for ID, lemmatized_text, text in zip(tqdm(list_of_IDs, desc="keyphrase extraction"), list_of_lemmatized_texts, list_of_texts):
        #     keyphrase_dict = {"TextID": ID, "Keyphrases":set()}
        #     for name, extractor in keyphrase_extractors.items():
        #         if name in groups_of_algorithms["textacy"]:
        #             list_of_keywords = extractor(text, language)
        #         else:
        #             list_of_keywords = extractor(lemmatized_text, language)
        #         keyphrase_dict["Keyphrases"].update(set(list_of_keywords))
        #     keyphrase_dict["Keyphrases"] = list(keyphrase_dict["Keyphrases"])
        #     keyphrase_table.upsert(keyphrase_dict, tinydb_where("TextID") == ID)

    db.close()

    print("End time: ", str(datetime.datetime.now()), flush=True)


def policy_detection_module():

    def load_keyphrases(db, language):
        list_of_lists_of_keyphrases = []
        print("Loading keyphrases in {}".format(language), flush=True)

        keyphrase_table = db.table("policies_keyphrases")
        list_of_keyphrase_dicts = keyphrase_table.all()

        language_table = db.table("policies_language")
        list_of_language_dicts = language_table.search(
            tinydb_where("DeterminedLanguage") == language
        )
        print("list_of_language_dicts: {}".format(len(list_of_language_dicts)), flush=True)

        list_of_language_IDs = [
            language_dict["TextID"] for language_dict in list_of_language_dicts
        ]

        del list_of_language_dicts

        list_of_keyphrase_dicts = [
            keyphrases_dict
            for keyphrases_dict in list_of_keyphrase_dicts
            if keyphrases_dict["TextID"] in list_of_language_IDs
        ]

        list_of_TextIDs = [
            keyphrase_dict["TextID"] for keyphrase_dict in list_of_keyphrase_dicts
        ]

        assert sorted(list_of_TextIDs) == sorted(list_of_language_IDs)
        del list_of_language_IDs

        policies_table = db.table("policies")
        list_of_policies_dicts = policies_table.all()
        list_of_URLs = [
            policy_dict["URL"]
            for policy_dict in list_of_policies_dicts
            if policy_dict["TextID"] in list_of_TextIDs
        ]

        del list_of_policies_dicts

        for keyphrase_dict in list_of_keyphrase_dicts:
            # list_of_keyphrases = (
            #     keyphrase_dict["MultiRake"]
            #     + keyphrase_dict["YakeOriginal"]
            #     + keyphrase_dict["PKE_TextRank"]
            #     + keyphrase_dict["PKE_SingleRank"]
            #     + keyphrase_dict["PKE_TopicRank"]
            #     + keyphrase_dict["PKE_PositionRank"]
            #     + keyphrase_dict["PKE_MultipartiteRank"]
            #     + keyphrase_dict["Textacy_sCAKE"]
            # )
            # list_of_lists_of_keyphrases.append(list_of_keyphrases)
            list_of_lists_of_keyphrases.append(keyphrase_dict["Keyphrases"])
            

        print(len(list_of_lists_of_keyphrases), flush=True)
        list_of_lists_of_keyphrases = [
            list(set([keyphrase.lower() for keyphrase in list_of_keyphrases]))
            for list_of_keyphrases in list_of_lists_of_keyphrases
        ]
        return list_of_TextIDs, list_of_URLs, list_of_lists_of_keyphrases



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
        print(number_of_policies + " policies:", Counter(all_keywords).most_common(50), flush=True)
        print("#unique keywords:", len(set(all_keywords)), flush=True)

        return list_of_dict_keyphrases

    def label_determination(
        language, list_of_dict_keyphrases, list_of_TextIDs, list_of_URLs
    ):

        print("Loading vectorizer and classifier", flush=True)
        vectorizer = load(
            "code/resources/trained_vectorizer_" + language + "_2023-01-22.pkl"
        )
        clf = load(
            "code/resources/VotingClassifier_soft_" + language + "_2023-01-22.pkl"
        )

        print("Vectorizer transformation", flush=True)
        X_unlabeled = vectorizer.transform(list_of_dict_keyphrases)

        print("Shape of unlabeled texts: {}".format(X_unlabeled.shape), flush=True)
        print("Predicting ...", flush=True)
        y_pred = clf.predict(X_unlabeled)
        y_pred_proba = clf.predict_proba(X_unlabeled)

        print(Counter(y_pred), flush=True)
        df = pd.DataFrame()
        df["TextID"] = list_of_TextIDs
        df["URL"] = list_of_URLs
        df["Language"] = language
        df["PredictedLabels"] = y_pred
        df_proba = pd.DataFrame(
            y_pred_proba, columns=["Probability_0", "Probability_1"]
        )
        df = pd.concat([df, df_proba], axis=1)

        Path("results/classification/").mkdir(parents=True, exist_ok=True)
        df.to_csv("results/classification/classification_" + language + "_" + crawl + ".csv")
        return df

    print("Start time of policy detection: ", str(datetime.datetime.now()), flush=True)

    storage = CachingMiddleware(JSONStorage)
    storage.WRITE_CACHE_SIZE = 25
    db = TinyDB(
        os.path.join(data_dir, "CPRA_policies_database_" + crawl + ".json"),
        storage=storage
    )
    label_table = db.table("policies_labels")

    list_of_languages = ["de", "en"]

    for language in tqdm(list_of_languages, unit="language"):
        print(f"Loading keyphrases of {language}", flush=True)
        (
            list_of_TextIDs,
            list_of_URLs,
            list_of_lists_of_keyphrases,
        ) = load_keyphrases(db, language)
        list_of_dict_keyphrases = keyphrase_analyzer(list_of_lists_of_keyphrases)

        if len(list_of_TextIDs) > 0:
            df_labels = label_determination(
                language, list_of_dict_keyphrases, list_of_TextIDs, list_of_URLs
            )
        else:
            print("No data for language", language, flush=True)
        list_of_dicts_with_label = df_labels.to_dict("records")
        del df_labels
        for dict_with_label in list_of_dicts_with_label:
            label_table.upsert(dict_with_label, tinydb_where("TextID") == dict_with_label["TextID"])

    db.close()
    print("End time: ", str(datetime.datetime.now()), flush=True)

if __name__ == "__main__":
    text_extraction_module()
    language_detection_module()
    keyphrase_extraction_module()
    policy_detection_module()
