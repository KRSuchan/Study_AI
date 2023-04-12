from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
raw_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    raw_data.data, raw_data.target, test_size=0.2, random_state=42, stratify=raw_data.target)

gnb_clf = GaussianNB()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
voting_clf = VotingClassifier(
    estimators=[('gnb', gnb_clf), ('rf', rnd_clf)], voting='soft')


for clf in (gnb_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
