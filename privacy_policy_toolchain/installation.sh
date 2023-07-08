#!/bin/bash
sudo apt update 
sudo apt install openjdk-8-jre
git clone https://github.com/ITSec-WWU-Munster/Unifying-Privacy-Policy-Detection.git
cd Unifying-Privacy-Policy-Detection
conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda create -y -n pptc python=3.7 pip libgcc-ng 
conda activate pptc
which python
conda install -y -c conda-forge icu 
pip install boilerpipe3==1.3 chardet=4.0.0 cld2-cffi=0.1.4 guess-language-spirit==0.5.3 jinja2=3.0.1 joblib jpype1 langid==1.1.6 multi-rake nltk numpy pandas pytorch=1.12.1 pycld2==0.41 pycld3==0.20 pymupdf==1.18.6 readabilipy==0.2.0 polyglot==16.7.4 scikit-learn=1.0.2 scipy setuptools spacy=3.4.1 spacy-lookups-data=0.3.0 textacy=0.11.0 tqdm html-sanitizer==1.9.1 markdownify==0.10.3 tinydb yake fasttext=0.9.2 langdetect unidecode=1.1.2 yaml pyicu==2.7.4 morfessor fuzzywuzzy
pip install git+https://github.com/boudinfl/pke.git
which python
python -m spacy download en_core_web_lg
python -m spacy download de_core_news_lg
python -m spacy download xx_ent_wiki_sm
python -c "import nltk; nltk.download('all')"
polyglot download LANG:de
polyglot download LANG:en
