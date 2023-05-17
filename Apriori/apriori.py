import pandas as pd
from apyori import apriori

store_data = pd.read_csv(
    "C:\GIT_KRSuchan\Study_AI\datasets\store_data\store_data.csv", header=None)

records = []
for i in range(0, 7501):
    records.append([str(store_data.values[i, j])for j in range(0, 20)])

association_rules = apriori(
    records, min_support=0.0045, min_confidence=0.6, min_lift=2)
association_results = list(association_rules)
print(len(association_results))

for item in association_results:
    pair = item[0]
    items = [x for x in pair]
    print("Rule : "+items[0]+"->"+items[1])
    print("Support : "+str(item[1]))
    print("Confidence : "+str(item[2][0][2]))
    print("Lift : " + str(item[2][0][3]))
    print("==================================")
