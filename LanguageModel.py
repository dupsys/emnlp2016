
# coding: utf-8

# In[1]:

from xlrd import * 
import re
def Load_Dictionary():
    #open_workbook('dictionary.xlsx')
    X = []
    wb = open_workbook('dictionary.xlsx')
    start = False
    for s in wb.sheets():
        for row in range(s.nrows):
                temp = []
                values = []

                for col in range(s.ncols):
                    values.append(s.cell(row,col).value)
                if type(values[0])==float:
                    values[0] = int(values[0])
                if type(values[1])==float:
                    values[1] = int(values[1])
                temp.append(values[0])
                temp.append(values[1])
                if temp[0]=='':
                    temp[0]=X[row-1][0]
                if temp[1]=='':
                    temp[1]=X[row-1][1]
                X.append(temp)
    X = X[1:]
    for x in range(len(X)):
        for y in range(2):
              X[x][y] = str(X[x][y])
    temp = []
    k = 0
    X_prime = X
    for i in range(len(X)):
        if(i>0):
            if(X[i][0]==X[i-1][0]):
                temp[k-1].append(X[i][1])
            else:
                k+=1
                temp.append(X[i])
    Dictionary = temp
    return Dictionary


# In[2]:

def LCS(X, Y):
    m = len(X)
    n = len(Y)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]: 
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    return C


# In[3]:

Dictionary = Load_Dictionary()
X = []
Y = []
for data in Dictionary:
    Y.append(data[1:])
    X.append(data[0])


# In[4]:

def Load_Twit_Data():
    Twits = []
    #with open("/home/melvin/LogLinear/Dataset_2016 for PhD/SMS_tweet/Mobile_tweet1.txt") as f:
    with open("tweetdata.txt") as f:
        for line in f:
            Twits.append(line)
    for i in range(len(Twits)):
        Twits[i] = Twits[i].replace('\r\n','')
    return Twits


# In[5]:

Twits = Load_Twit_Data()


# In[6]:

import numpy as np
def Preprocess_Twits_and_Dictionary(X):
    #convert all string to lower case.
    for i in range(len(X)):
        X[i] = X[i].lower()
    for i in range(len(Twits)):
        Twits[i] = Twits[i].lower()
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            Y[i][j] = Y[i][j].lower()
    #Removal of panctuations.
    for i in range(len(Twits)):
        Twits[i] = Twits[i].replace('</s>','')
        Twits[i] = Twits[i].replace('<s>','')
    import string
    for i in range(len(Twits)):
        for c in string.punctuation:
            Twits[i] = Twits[i].replace(c," ")
    for i in range(len(Twits)):
        Twits[i] = '<s> '+Twits[i]+' </s>'
    for m in range(10): 
        for i in range(len(Twits)):
            Twits[i] = Twits[i].replace('  ',' ')    
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            for c in string.punctuation:
                Y[i][j] = Y[i][j].replace(c," ")
    #Removal of multiple spaces.
    for m in range(10): 
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                for c in string.punctuation:
                    Y[i][j] = Y[i][j].replace('  ',' ')            
    Preprocessed_Twits = Twits
    return Preprocessed_Twits, Dictionary


# In[7]:

Preprocessed_Twits, Dictionary = Preprocess_Twits_and_Dictionary(X)


# In[8]:

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


# In[9]:

def levenshtein_ratio(s1, s2):
    Ldist = levenshtein(s1, s2)
    return float((len( s1 )+len( s2 ))-Ldist) / float(len( s1 )+len( s2 ))


# In[10]:

def count_bi_gram(X,token):
    counts = sum([row.count(token) for row in X])
    return counts
def count_unigram(X,token):
    counts = sum([row.count(token) for row in X])
    return counts


# In[11]:

def Create_target_keys(Y):
    k = 0
    Key_Dictionary = []
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            temp = [k,Y[i][j]]
            Key_Dictionary.append(temp)
            k+=1
    Key_Dictionary = {Key_Dictionary[i][1]: Key_Dictionary[i][0] for i in range(0, len(Key_Dictionary))}
    return Key_Dictionary


# In[12]:

Key_Dictionary = Create_target_keys(Y)


# In[13]:

def Uni_grams(X):
    Uni = []
    for i in X:
        temp = i.split(" ")
        Uni.append(temp)
    return Uni


# In[14]:

def Bi_grams(X):
    bi_grams = []
    Uni = Uni_grams(X)
    for i in range(len(Uni)):
        temp = []
        for j in range(len(Uni[i])-2):
            bi = Uni[i][j]+" "+Uni[i][j+1]
            temp.append(bi)
        bi_grams.append(temp)
    return bi_grams


# In[15]:

bi_grams = Bi_grams(Twits)


# In[16]:

def Conditional_Probability1(c1,c2,Twits):
    Uni = Uni_grams(Twits)
    num = count_bi_gram(Uni,c2)
    den = count_unigram(Uni,c1)
    probability = float(num)/float(den)
    return probability


# In[17]:

def backTrack(C, X, Y, i, j):
    if i == 0 or j == 0:
        return ""
    elif X[i-1] == Y[j-1]:
        return backTrack(C, X, Y, i-1, j-1) + X[i-1]
    else:
        if C[i][j-1] > C[i-1][j]:
            return backTrack(C, X, Y, i, j-1)
        else:
            return backTrack(C, X, Y, i-1, j)


# In[18]:

def backTrackAll(C, X, Y, i, j):
    if i == 0 or j == 0:
        return set([""])
    elif X[i-1] == Y[j-1]:
        return set([Z + X[i-1] for Z in backTrackAll(C, X, Y, i-1, j-1)])
    else:
        R = set()
        if C[i][j-1] >= C[i-1][j]:
            R.update(backTrackAll(C, X, Y, i, j-1))
        if C[i-1][j] >= C[i][j-1]:
            R.update(backTrackAll(C, X, Y, i-1, j))
        return R


# In[19]:

x = "AATCC"
y = "ACACG"
m = len(x)
n = len(y)
C = LCS(x, y)
print "Some LCS: '%s'" % backTrack(C, x, y, m, n)
print "All LCSs: %s" % backTrackAll(C, x, y, m, n)


# In[20]:

def LCS_Ratio(X, Y):
    m = len(X)
    n = len(Y)
    C = LCS(X, Y)
    ratio = float(len(backTrack(C, X, Y, m, n)))/float(m)
    return ratio


# In[21]:

def Similarity_measure(C2,Target):
    similarity = []
    #print C2, Target
    for i in Target:
        LCSRatio = LCS_Ratio(i, C2)
        similarity.append([LCSRatio,i])
    return similarity
            


# In[22]:

def Similarity_measure_Testing(C2,Target):
    similarity = []
    LCSRatio = LCS_Ratio(Target, C2)    
    return LCSRatio


# In[23]:

#def Maximum_Probability(C2,similarity,probability,temp,Key_Dictionary):
def Maximum_Probability(C2,similarity,probability,Key_Dictionary):
    maximum_probability = 0
    target = ""
    Vs = C2
    for j in range(len(similarity)):
        #answer = (similarity[j][0] + probability + temp[j])/float(3)
        answer = (similarity[j][0] + probability)/float(2)
        if answer>maximum_probability:
            maximum_probability = answer
            target = similarity[j][1]
            #Vs_features = [similarity[j][0],probability,temp[j]]
            Vs_features = [similarity[j][0],probability]
            Target_Key = Key_Dictionary[target]
            Vt = target
    return Vs,Vt,Vs_features,Target_Key
    


# In[24]:

def feature_vector_training(Twits,bi_grams,Key_Dictionary,Dictionary,X,Y):
    Vs = []
    temp = []
    Feature_vector = []
    New_Twits = []
    Y_Target = []
    Reconstructed_twit = []
    Training_abrivitions = []
    k = 0
    for bigrams in bi_grams:
        Reconstructed_twit = ['<s>']
        temporary_features = []
        for i in range(len(bigrams)):
            C = bigrams[i].split(" ")
            C1 = C[0]
            C2 = C[1]
            if C2 in X:
                temp = []
                position = X.index(C2)
                similarity = Similarity_measure(C2, Y[position])
                probability = Conditional_Probability1(C1,C2,Twits)
                Vs,Vt,Vs_features,Target_Key = Maximum_Probability(C2,similarity,probability,Key_Dictionary)
                temp = []
                Feature_vector.append(Vs_features)
                Y_Target.append(Target_Key)
                Reconstructed_twit.append(Vt)
                Training_abrivitions.append(C2)
            else:
                C = bigrams[i].split(" ")
                Reconstructed_twit.append(C[1])
        Reconstructed_twit.append('</s>')
        New_Twits.append(" ".join(Reconstructed_twit))
        print "####################################################################################"
        print Twits[k]
        print " ".join(Reconstructed_twit)
        k+=1
    return Feature_vector,Y_Target, Training_abrivitions


# In[25]:

Feature_vector,Y_Target,Training_abrivitions = feature_vector_training(Twits,bi_grams,Key_Dictionary,Dictionary,X,Y)


# In[26]:

print Feature_vector
print Y_Target


# In[27]:

def stochastic_gradient_descent(x, y, iters):
    costs = []
    m = y.size # number of data points
    theta = np.random.rand(2) # random start
    alpha = float(0.01)
    history = [theta] # to store all thetas
    preds = []
    for i in range(iters):
        if i>0:
            alphap = alpha
            alpha = float(alphap)/float(1+(float(i)/float(len(x))))
           
        pred = np.dot(x, theta)
        error = pred - y 
        cost = np.sum(error ** 2) / (2 * m)
        costs.append(cost)
        
        if i % 25 == 0: preds.append(pred)

        gradient = x.T.dot(error)/m 
        theta = theta - alpha * gradient  # update
        #history.append(theta)
    print theta
    print "cost = "+str(cost)
    return history, costs, preds


# In[28]:

iters = 10 # set number of iterations
history, cost, preds = stochastic_gradient_descent(np.array(Feature_vector), np.array(Y_Target), iters)
theta = history[-1]


# In[29]:

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.n_iter = 5
clf.fit(np.array(Feature_vector), np.array(Y_Target))


# In[30]:

print "pridicted = " + str(clf.predict([Feature_vector[0]]))
print "Actual = " + str(Y_Target[0])


# In[31]:

lf = SGDClassifier(loss="log")
lf.n_iter = 5000
lf.fit(np.array(Feature_vector),  np.array(Y_Target))
lf.predict_proba(Feature_vector[0])                      


# In[32]:

print "pridicted = " + str(lf.predict([Feature_vector[2]]))
print "Actual = " + str(Y_Target[2])


# In[33]:

tweet = "<s> k ya i hav to finish </s>"
temp = tweet.split(" ")
for i in range(1,len(temp)-1):
    for j in range(len(Y)):
        bigram = temp[i-1]+" "+temp[i]
        f1 = bigram
        f2 = Similarity_measure(temp[i], Y[j])


# In[34]:

lf.predict_proba(Feature_vector[0])


# In[35]:

print Key_Dictionary.values()


# In[36]:

List_classes = lf.classes_
print List_classes


# In[37]:

List_probabilities = lf.predict_proba(Feature_vector[0])[0]


# In[38]:

class_max_prob = List_classes[List_probabilities.argmax()]
print class_max_prob
List_classes = List_classes.tolist()


# In[39]:

def Testing_Feature_Vector(C1,C2,lf):
    l = 0
    target = ""
    target_feature_vector = []
    for key in Key_Dictionary:
        if Key_Dictionary[key] in List_classes:
            similarity = Similarity_measure_Testing(C2, key)
            the_probability = Conditional_Probability1(C1,C2,Twits)
            the_feature_vector = [the_probability, similarity]
            position = List_classes.index(Key_Dictionary[key])
            List_probabilities = lf.predict_proba(Feature_vector[0])[0]
            if l==0:
                maximum_probability = List_probabilities[position]
                l = l + 1
            else:
                if maximum_probability < List_probabilities[position]:
                    maximum_probability = List_probabilities[position]
                    target = key
                    target_feature_vector = the_feature_vector
    return C2, target, maximum_probability, target_feature_vector


# In[40]:

for tweet in Twits:
    temp = tweet.split(' ')
    print tweet
    new_tweet = '<s>'
    for count in range(len(temp)-2):
        C1 = temp[count]
        C2 = temp[count+1]
        if C2 in Training_abrivitions:
            abriviation, target, maximum_probability, target_feature_vector = Testing_Feature_Vector(C1,C2,lf)
            new_tweet = new_tweet + " " + target
            print abriviation,target, maximum_probability, target_feature_vector
        else:
            new_tweet = new_tweet + " " + C2
        
    new_tweet = new_tweet + " </s>"
    print new_tweet
    


# In[41]:


def flatten(x):
    '''
    Returns a single, flat list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables).
    '''
    result = []
    for el in x:
        #if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


# In[42]:

def intersect(a, b):
     return list(set(a) & set(b))


# In[43]:

def Unigrams(Twits):
    Uni_grams = []
    for i in Twits:
        Uni_grams.append(i.split(" ")) 
    return Uni_grams


# In[44]:

Uni_grams = Unigrams(Twits)


# In[45]:

Training_abreviations = intersect(flatten(X), flatten(Uni_grams))
print Training_abreviations


# In[46]:

def count_unigram(X,token):
    counts = sum([row.count(token) for row in X])
    return counts


# In[ ]:

Abriviation_frequencies = []
for i in Training_abreviations:
    Abriviation_frequencies.append(count_unigram(Uni_grams,i))
print Abriviation_frequencies


# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
'''
x = np.array([0,1,2,3])
y = np.array([20,21,22,23])
my_xticks = ['John','Arnold','Mavis','Matt']
'''
x = []
for i in range(1,len(Abriviation_frequencies)+1):
    x.append(i)
x = np.array(x)
plt.xticks(x,Training_abreviations)
plt.bar(x, Abriviation_frequencies)
plt.show()


# In[ ]:



