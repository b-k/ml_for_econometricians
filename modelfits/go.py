# As with most structured code, the main procedure is at the bottom, so this
# may best be read bottom-up.
# Run using python 3, not 2.7

import math
import sqlite3
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid

# just fit the data -- no test/train
# comparative statics
# Bootstrap
# test/train, w/confusion matrices

# predict Belize et al
# entropy, frowny face

cursor = sqlite3.connect('remit.db').cursor()

def QQ(query):
    cursor.execute(query)
    return np.array(cursor.fetchall())

def avg(x):
    return sum(x)/len(x)

def stddev(x):
    return math.sqrt(avg((x - avg(x))*(x - avg(x)))/ (len(x)-1))

def fit_predict(model, dependent, independent):
        estimate = model.fit(X=dependent, y=independent.ravel())
        guess = estimate.predict(dependent)
        return estimate, guess

def get_data(query):
    L = QQ("select has_net_in from dataset")
    R = QQ("select " + query + " from dataset")
    return L, R

def make_boot_sample (L, R):
    L_out = L.copy()
    R_out = R.copy()
    ll=L.shape[0]
    while True:
      for i in range(ll):
        j = np.random.randint(0, ll)
        L_out[i] = L[j]
        R_out[i, :] = R[j, :]
      if (sum(L_out)!=ll or sum(L_out)!=0): break  #bad draw
    return L_out, R_out

def bootstrap(m, query, fn):
    L,R= get_data(query)
    stats = []
    for _ in range(50):
        Lb, Rb = make_boot_sample(L, R)
        stats.append(fn(m, Lb, Rb))
    return avg(stats),stddev(stats)

def comparative_statics_core(m, L, R, offset=.1):
    for i in range(R.shape[1]):
        R[:,i]/=avg(R[:,i]) #darn it, scikit.

    m, guess1 = fit_predict(m, R, L)
    guesses1 = sum(abs(guess1))

    R[:,0] *=  (1+offset)
    guess2 = m.predict(R)
    guesses2 = sum(abs(guess2))

    return (guesses2 - guesses1)/len(L)

def comparative_statics(m, query, offset=.1):
    L, R=get_data(query)
    return comparative_statics_core(m, L, R, offset)

def test_train_core(L, R, m):
    guesses=[]
    losses=[]
    for _ in range(20): #not quite a bootstrap; forgive me
        indices = np.random.permutation(len(L))
        L_train = L[indices[:-100]].ravel()
        R_train = R[indices[:-100]]
        L_test  = L[indices[100:]]
        R_test  = R[indices[100:]]

        _, guess = fit_predict(m, R_test, L_test)
        #diff =  guess - L_test.ravel()
        fusion = metrics.confusion_matrix(L_test.ravel(), guess)
        losses.append(fusion[0][0]/sum(fusion[:,0]))
        #guesses.append(sum(abs(guess)))
    #return avg(guesses), stddev(guesses) * 2 
    return avg(losses), stddev(losses) * 2 

def test_train(m, query):
    L, R=get_data(query)
    return test_train_core(L, R, m)

def make_guesses():
    guesses=[]
    L, R = get_data(all_vars)

    print("\t\tData", end='')
    for model in models:
        print("\t%s" %(model[1],),end='')
    print()

    for ctry in [
            ["United States", 1],
            ["China", 0],
            ["Ecuador", 0],
            ["Malta", 0],
            ["Iceland", 0],
            ["Belize", 1],
            ["Luxembourg", 1],
            ["Congo, Rep.", 1]]:
      Rbb=QQ("select " + all_vars + " from dataset where name='" + ctry[0] + "'")
      print(ctry[0]+ "\t", end='')
      if (ctry[0]in ("Ecuador", "Iceland", "Belize","China", "Malta")): print("\t", end='')
      print(str(ctry[1])+ ":\t", end='')
      for model in models:

        for _ in range(50):
            Lb, Rb = make_boot_sample(L, R)
            estimate = model[0].fit(X=Rb, y=Lb.ravel())
            guesses.append(estimate.predict(Rbb)[0])

        #guess = estimate.predict(Rb)
        print("%.2f ±(%.2f)\t" % (avg(guesses),2*stddev(guesses)), end='')
      print()

np.random.seed(0)

from sklearn.neural_network import MLPClassifier

models = [[linear_model.LogisticRegression(), "Logit"],
         [svm.SVC(), "SVM"],
         [NearestCentroid(), "Centroid"]]



all_vars= """pop, log_gdp, electricity_access, exports_per_gdp, urban_pop,
               journal_publications/pop, net_migration, ln_surface_area_sq_km, urban_pop """

def loop_over_models(want_boot=0, want_tt=0):
  for query in [
    all_vars,
    """gdp""",
    """exports_per_gdp""",
    """log_gdp""",
    """electricity_access""",
    """journal_publications""",
    """ln_surface_area_sq_km""",
    """log_gdp, journal_publications""",
    """journal_publications, log_gdp""",
    ]:
    print()
    print(query)
    for model in models:
        if want_tt:
            ta, ts = test_train(model[0], query)
            print("%+.3f ±%.3f: %s" %(ta, ts,model[1]))
        if want_boot:
            ba, bs = bootstrap(model[0], query, comparative_statics_core)
            print("%+.3f ±%.3f: %s" %(ba, bs,model[1]))
        if not want_tt and not want_boot:
            #error = get_boot(model[0], query)
            print("%+.3f: %s" %(comparative_statics(model[0], query), model[1]))


def correlations():
    print("has_net_in, pop, log_gdp, net_migration, journal_publications/cap,electricity_access, surface_area_sq_km")
    print(np.corrcoef(QQ("select has_net_in, pop, log_gdp, net_migration, journal_publications/pop,electricity_access, surface_area_sq_km from dataset").transpose()))

########## int main

print("""
━━━━━━━━━━━━━━━ Some correlations.""")
correlations()
print("net_migration, GDP")
print(np.corrcoef(QQ("select net_migration, gdp from dataset").transpose())[0][1])

print("Are GDP and Surface area correlated?")
print(np.corrcoef(QQ("select gdp, ln_surface_area_sq_km from dataset").transpose())[0][1])

print("net migra, Surface area correlated?")
print(np.corrcoef(QQ("select net_migration, ln_surface_area_sq_km from dataset").transpose())[0][1])

print("net migra, Surface area correlated?")
print("select net_migration, journal_publications/pop,electricity_access,pop, log_gdp, ln_surface_area_sq_km from dataset")
print(np.corrcoef(QQ("select net_migration, journal_publications/pop,electricity_access,pop, log_gdp, surface_area_sq_km from dataset").transpose()))

print("""
━━━━━━━━━━━━━━━ comparative statics.""")
loop_over_models()

print("""
━━━━━━━━━━━━━━━ boostrap comparative statics.""")
loop_over_models(want_boot=1)

print("""
━━━━━━━━━━━━━━━ Test/Train validation: true positives/all positives.""")
loop_over_models(want_tt=1)

print("""
━━━━━━━━━━━━━━━ Guesses.""")
make_guesses()
