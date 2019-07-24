from model import KNN
from pprint import pprint

def main():
    k = 3
    split = 0.8
    header, x_train, y_train, x_test, y_test = KNN.load('iris.csv', split)
    knn = KNN(x_train, y_train, k)
    y_pred = knn.test(x_test)
    pprint(y_pred)
    accuracy = KNN.accuracy(y_pred, y_test)
    print(f"Accuracy est de: {accuracy}")

    flower = [0.8,0.0,5,0]
    prediction = knn.test([flower])
    print(f"La fleur {flower} est un {prediction[0]}")

if __name__ == "__main__":
    main()