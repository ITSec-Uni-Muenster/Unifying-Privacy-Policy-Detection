# Unifying Privacy Policy Toolchain

This is the accompanying repository for the "Unifying Privacy Policy Detection" paper which was published in the [Privacy Enhancing Technology Symposium (PETS) 2021](https://petsymposium.org/2021/paperlist.php).

The aim of this project is to support privacy policy researchers with a unified solution for creating privacy policy corpora. 

At the moment, we have uploaded the source code as a proof of concept, according with the trained classifiers and vectorizers in English and German. However, we are aiming to provide a pip package as soon as possible in order to ease the application of this toolchain. 

## Explanation
The toolchain consists of five steps:
1. Finding potential privacy/cookie policies on websites
2. Text-from-HTML extraction
3. Language detection
4. Key phrase extraction
5. Classification

Currently, the tool saves the data and according metadata in a [TinyDB](https://tinydb.readthedocs.io/en/latest/) database, which offers benefits such as potability and simplicity. If a different data storage format is required, the code can be adapted easily due to its modular design. 

## Structure of the repository
The current structure of the repository is depicted as follows:

```
.
|-- README.md
`-- privacy_policy_toolchain
    |-- feature_list
    |   |-- feature_list_de.txt
    |   `-- feature_list_en.txt
    |-- ppt.py
    `-- resources
        |-- VotingClassifier_soft_de.pkl
        |-- VotingClassifier_soft_en.pkl
        |-- trained_vectorizer_de.pkl
        `-- trained_vectorizer_en.pkl
```



