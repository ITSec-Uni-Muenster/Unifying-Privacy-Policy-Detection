# Unifying Privacy Policy Detection Toolchain

This is the accompanying repository for the "Unifying Privacy Policy Detection" paper published in the [Pri­va­cy En­han­cing Tech­no­lo­gies Sym­po­si­um (PETS) 2021](https://petsymposium.org/2021/paperlist.php).

The aim of this project is to support privacy policy researchers with a unified solution for creating privacy policy corpora based on currently available best-practices. 

At the moment, we have uploaded the source code as a proof of concept, according with the trained classifiers and vectorizers in English and German. We are planning to provide a pip package as soon as possible in order to ease the application of this toolchain. 

## Explanation
The toolchain consists of five steps:
1. Finding potential privacy/cookie policies on websites
2. Text-from-HTML extraction
3. Language detection
4. Key phrase extraction
5. Classification


## Structure of the repository
The current structure of the repository is depicted as follows:

```
.
|-- LICENSE
|-- README.md
|-- privacy_policy_link_detection
|   |-- README.md
|   |-- custom_command_find_privacy_policies.py
|   `-- demo_privacy_policy_download.py
`-- privacy_policy_toolchain
    |-- code
    |   |-- ppt.py
    |   `-- resources
    |       |-- VotingClassifier_soft_de.pkl
    |       |-- VotingClassifier_soft_en.pkl
    |       |-- trained_vectorizer_de.pkl
    |       `-- trained_vectorizer_en.pkl
    |-- data
    |   `-- privacy_policies
    |-- environment.yml
    |-- feature_list
    |   |-- feature_list_de.txt
    |   `-- feature_list_en.txt
    |-- logs
    |   `-- language_analysis
    `-- results
        `-- classification
```

The folder `resources` contains the trained models and the vectorizers for both English and German.

## Paper
Henry Hosseini, Martin Degeling, Christine Utz, Thomas Hupperich. "Unifying Privacy Policy Detection." PETS 2021.

## Contact
* Henry Hosseini: henry.hosseini@wi.uni-muenster.de


