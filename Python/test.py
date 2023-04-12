# 학습 단계
train_data_input = read_train_data_input()
train_data_target = read_train_data_target()
classifier = DecisionTreeClassifier()
classifier.fit(train_data_input, train_data_target)
# 예측 단계
a = input("a의 값 ?")
b = input("b의 값 ?")
y = classifier.predict([[a, b]])
print(y)
