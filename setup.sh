mkdir -p /home/appuser/.nltk_data/corpora
wget -q -O /home/appuser/.nltk_data/corpora/stopwords.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip
unzip -q /home/appuser/.nltk_data/corpora/stopwords.zip -d /home/appuser/.nltk_data/corpora/
