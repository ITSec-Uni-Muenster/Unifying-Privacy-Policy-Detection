## Environment Setup


We use [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) on Ubuntu 20.04 in order to create the Python environment. Please ensure that you have Java (e.g. openjdk-8-jre) installed. To setup the environment, please follow the instructions as provided below:

```
conda create -n pptc python=3.6 pip
conda activate pptc
conda env update -f environment.yml

```

For spacy, we need to install the necessary models:

```
python -m spacy download en_core_web_lg
python -m spacy download de_core_news_lg
python -m spacy download xx_ent_wiki_sm
```

The tool expects a path like `data` followed by a folder named with, e.g., the date of the crawl, which contains the rawl HTML/XML files of the respecting crawl. The tool reads these files and saves the data and according metadata in a [TinyDB](https://tinydb.readthedocs.io/en/latest/) database, which offers benefits such as potability and simplicity. If a different data storage format is required, the code can be adapted easily due to its modular design. If you have any questions about how to work with the code or regarding replication of our results, feel free to contact us.
