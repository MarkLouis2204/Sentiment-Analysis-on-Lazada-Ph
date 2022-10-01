from ntpath import join
from sre_parse import Tokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import stopwordsiso as stopwords
nltk.download('stopwords')


xtrain=["ang ganda ng product!",
"lupet nito",
"ang bilis ng delivery",
"ang panget",
"di na ako bibili dito",
"sayang ang pera ang bagal"]

xtest=["ganda ng product!"]

ytrain=[1,1,1,0,0,0] #1=positve 0=negative

tokenizer= RegexpTokenizer(r'\w+')
tlen_stopwords = stopwords.stopwords(["tl","en"])
ps = PorterStemmer()

def getcleanedtext(text):
    text=text.lower()

    tokens= tokenizer.tokenize(text)
    new_tokens = [token for token in tokens if token not in tlen_stopwords]

    stemmed_tokens= [ps.stem(tokens)for tokens in new_tokens]

    clean_text = "".join(stemmed_tokens)
    return clean_text

xclean=[getcleanedtext(i) for i in xtrain]
xtclean=[getcleanedtext(i) for i in xtest]

cv = CountVectorizer(ngram_range=(1,2))

x_vec=cv.fit_transform(xclean).toarray()

xt_vect=cv.transform(xtclean).toarray()

mn = MultinomialNB()
mn.fit(x_vec, ytrain)
y_pred=mn.predict(xt_vect)
print(xclean)
print(xtclean)
print(y_pred)