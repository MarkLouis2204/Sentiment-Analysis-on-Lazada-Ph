import csv
from itertools import count
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

def getcleanedtext(text):
    text=text.lower()

    tokens= tokenizer.tokenize(text)
    new_tokens = [token for token in tokens if token not in tlen_stopwords]

    stemmed_tokens= [ps.stem(tokens)for tokens in new_tokens]

    clean_text = " ".join(stemmed_tokens)
    return clean_text

with open(r"C:\Users\zacha\Desktop\research\try1.csv",encoding="utf8") as csvfile:
    reader=csv.DictReader(csvfile)
    count = 0;
    xtrain=[]
    ytrain=[]

    for row in reader:
        count = count +1
        # print (row['comment'])
        xtrain.append(row['comment'])
        ytrain.append(row['label'])
        if count > 10:
            break

with open(r"C:\Users\zacha\Desktop\research\test1.csv",encoding="utf8") as csvfile:
    reader=csv.DictReader(csvfile)
    count = 0;
    xtest=[]

    for row in reader:
        count = count +1
        # print (row['comment'])
        xtest.append(row['comment']) ........
        if count > 10:
            break
# print(xtrain)
# print(ytrain)

tokenizer= RegexpTokenizer(r'\w+')
tlen_stopwords = stopwords.stopwords(["tl","en"])
ps = PorterStemmer()

xclean=[getcleanedtext(i) for i in xtrain]
xtclean=[getcleanedtext(i) for i in xtest]
print(xclean)
cv = CountVectorizer(ngram_range=(1,2))

x_vec=cv.fit_transform(xclean).toarray()

xt_vect=cv.transform(xtclean).toarray()

mn = MultinomialNB()
mn.fit(x_vec, ytrain)
y_pred=mn.predict(xt_vect)
print(xclean)
print(xtclean)
print(y_pred)