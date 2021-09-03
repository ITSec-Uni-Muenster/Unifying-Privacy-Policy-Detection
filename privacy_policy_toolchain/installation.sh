#!/bin/bash
sudo apt update 
sudo apt install openjdk-8-jre
git clone https://github.com/ITSec-WWU-Munster/Unifying-Privacy-Policy-Detection.git
cd Unifying-Privacy-Policy-Detection
conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda create -y -n pptc python=3.6 pip
conda activate pptc
which python
conda install -y -c conda-forge chardet=4.0.0 cld2-cffi=0.1.4 fuzzywuzzy icu jinja2=2.11.2 joblib jpype1 nltk numpy pandas pytorch=1.7.1 scikit-learn=0.24.0 scipy setuptools spacy=2.3.5 spacy-lookups-data=0.3.0 textacy=0.10.0 tika=1.24 tqdm=4.61.0 unidecode=1.1.2 yaml tinydb yake fasttext=0.9.2 langdetect pyicu==2.6 morfessor
pip install boilerpipe3==1.3 guess-language-spirit==0.5.3 langid==1.1.6 multi-rake pycld2==0.41 pycld3==0.20 pymupdf==1.18.6 readabilipy==0.2.0 polyglot==16.7.4
which python
pip install git+https://github.com/boudinfl/pke.git
python -m spacy download en_core_web_lg
python -m spacy download de_core_news_lg
python -m spacy download xx_ent_wiki_sm
python -c "import nltk; nltk.download('all')"
polyglot download LANG:de
polyglot download LANG:en