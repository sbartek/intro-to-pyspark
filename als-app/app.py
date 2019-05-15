from flask import Flask
import pandas as pd

app = Flask(__name__)
data_df = pd.read_csv("data/prediction/part-00000-fc0f4323-1d9a-4ddb-a4b1-d069eb0b2760-c000.csv")

@app.route("/predict/<user_id>")
def predict(user_id):
    print(data_df)
    user_prediction = data_df.query(f"user_idx == '{user_id}'")
    products = list(user_prediction['product_idx'])
    return ",".join([str(product) for product in products])
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1234)
