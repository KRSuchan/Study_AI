from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data = ["a", "a", "b", "c", "c"]
# encoder.fit(data)
# encoded_data = encoder.transform(data)
encoded_data = encoder.fit_transform(data)
print(encoded_data)
labeled_data = encoder.inverse_transform(encoded_data)
print(labeled_data)
