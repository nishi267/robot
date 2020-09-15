import pickle
from io import StringIO

from flask import Flask, request, make_response, Response, send_file

from flasgger import Swagger
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle

from sklearn.cluster import KMeans


app = Flask(__name__)
Swagger(app)

pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)
picky_in = open("cluster.pkl", "rb")
clusts = pickle.load(picky_in)


@app.route('/')
def welcome():
    return "Welcome All"


@app.route('/predict_file', methods=["POST"])
def predict_note_file1():
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true

    responses:
        200:
            description: The output values

    """
    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction = classifier.predict(df_test)

    return str(list(prediction))


@app.route('/predict_similar', methods=["POST"])
def predict_note_file():
    """Let's Cluster the Test cases for similarity Level
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true

    responses:
        200:
            description: The output values

    """
    df_test = pd.read_csv(request.files.get("file"), encoding='unicode_escape')
    df_test = df_test.dropna(axis=0, how='any')
    df_test['combine6'] = df_test.iloc[:, 1] + df_test.iloc[:, 2] + df_test.iloc[:, 3]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
    vec.fit(df_test.combine6.values)
    features = vec.transform(df_test.combine6.values)

    clustr = KMeans(init='k-means++', n_clusters=5, n_init=10)
    clustr.fit(features)
    df_test['cluster_labels'] = clustr.labels_
    output = StringIO()
    df_test.to_csv(output)
    return Response(output.getvalue(), mimetype="text/csv")
    # return "Check the file is generated"
    # --resp = make_response(df_test.to_csv())
    # resp.headers["Content-Disposition"] = ("attachment; filename=%s" % filename)
    # resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
    # --resp.headers["Content-Type"] = "text/csv"
    # resp.headers["Content-Disposition"] = ("attachment; filename=%s" % filename)
    # --return resp
    # & buffer = StringIO()
    # & df_test.to_csv(buffer, encoding='utf-8')
    # & buffer.seek(0)
    # & return send_file(buffer, attachment_filename="test.csv", mimetype='text/csv')


# return make_response(df_test.to_csv(), mimetype="text/csv")


if __name__ == '__main__':
    app.run()
