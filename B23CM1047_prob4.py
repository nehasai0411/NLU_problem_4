import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


def load_bbc_data(csv_path="bbc_data.csv"):
    df = pd.read_csv(csv_path)

    # Keep only sport and politics
    df = df[df["labels"].isin(["sport", "politics"])].copy()

    # Convert to required format
    texts = df["data"].astype(str).str.lower().tolist()
    labels = df["labels"].str.upper().tolist()   # SPORT / POLITICS

    return texts, labels



def train_and_evaluate(vectorizer, model, texts, labels, title):

    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\n=====", title, "=====")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))



def main():
    texts, labels = load_bbc_data("bbc_data.csv")

    print("Total samples:", len(texts))
    print("Classes:", set(labels))

   
    bow = CountVectorizer()
    nb = MultinomialNB()
    train_and_evaluate(bow, nb, texts, labels, "BoW + Naive Bayes")

    
    tfidf = TfidfVectorizer()
    lr = LogisticRegression(max_iter=2000)
    train_and_evaluate(tfidf, lr, texts, labels, "TF-IDF + Logistic Regression")

    ngram = TfidfVectorizer(ngram_range=(1, 2))
    svm = LinearSVC()
    train_and_evaluate(ngram, svm, texts, labels, "TF-IDF (n-grams) + Linear SVM")


if __name__ == "__main__":
    main()
