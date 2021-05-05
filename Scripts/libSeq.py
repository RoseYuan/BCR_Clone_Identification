import Levenshtein
#install with pip install python-Levenshtein

    
def Normalized_Hamming_dist(string1,string2):
    """Returns the Hamming distance between array A and B
    if A and B are of different length, the distance is 1"""
    if len(string1) != len(string2):
        return 1
    else:
        H = 0
        for i in range(len(string1)):
            if string1[i] != string2[i]:
                H += 1     
        return H/len(string1)


def Normalized_Levenshtein_dist(string1, string2):
    """
    [ref] Yujian, Li, and Liu Bo. "A normalized Levenshtein distance metric." 
    IEEE transactions on pattern analysis and machine intelligence 29.6 (2007): 1091-1095.
    """
    if len(string1)==0 and len(string2)==0:
        return 1
    else:
        Lev = Levenshtein.distance(string1, string2)
        norm_lev = 2*Lev/(len(string1)+len(string2)+Lev)
        
        return norm_lev


def Cosine_dist(tf_idf1:dict,tf_idf2:dict):
    if tf_idf1.keys() != tf_idf2.keys():
        raise ValueError("different set of keys! Cannot compute the inner product.")
    return 1-sum([tf_idf1[k]*tf_idf2[k] for k in tf_idf1.keys()])


if __name__ == "__main__":
    
    strA = "Hello world"
    strB = "Hecko korld"
    strC = "Helllo world"
    
    print(Normalized_Hamming_dist(strA,strB))
    print(Normalized_Levenshtein_dist(strA,strB))
    print(Normalized_Levenshtein_dist(strA,strC))


    