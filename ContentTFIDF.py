from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords

class ContentTFIDF:
    
    def __init__(self, data):
        self.data = data

    def calculateTFIDF(self):
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2) ,stop_words=stopwords.words('english'))
        tfidf_content = tfidf.fit_transform(self.data['genre'])
        return tfidf_content