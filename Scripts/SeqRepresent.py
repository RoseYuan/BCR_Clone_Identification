from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import pandas as pd

"""
Compute tf-idf representation:
"""

def tf_idf_BCR(Xs, Ys=None, k=4, l=130):
    truncated_X = [x[-l:] for x in Xs]
    len_X = len(truncated_X)
    if Ys is not None:
        truncated_Y = [y[-l:] for y in Ys]
        truncated_X = truncated_X + truncated_Y
    tf_idf, idf = cal_tf_idf(truncated_X, k=k)
    if Ys is not None:
        tf_idf_Xs = tf_idf.iloc[:len_X,:]
        tf_idf_Ys = tf_idf.iloc[len_X:,:]
    else:
        tf_idf_Xs = tf_idf
        tf_idf_Ys = None 
    return tf_idf_Xs, tf_idf_Ys, idf

def cal_tf_idf(sequences, k):
    textword = [getKmers(x,k=k) for x in sequences]
    for i in range(len(textword)):
        textword[i] = ' '.join(textword[i])
    vec = TFIDF()
    # Default L2 regularization
    tfidf = vec.fit(textword)
    X1 = tfidf.transform(textword)
    # Produce tf-idf matrix
    kmers = tfidf.get_feature_names()
    idf = tfidf.idf_
    return pd.DataFrame(X1.toarray(), columns=kmers,dtype='float32'), idf

def getKmers(sequence:str, k=6):
    """
    Break DNA into kmer and process into word format.
    :param sequence: a DNA sequence
    :return:
    """
    # including 'n' increases a lot of memory usage: 5^k - 4^k
    return [sequence[x:x+k].lower() for x in range(len(sequence)-k+1) if 'n' not in sequence[x:x+k].lower()]

