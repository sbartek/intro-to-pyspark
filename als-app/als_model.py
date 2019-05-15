import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS, ALSModel


class ALSRecModel:

    def __init__(self, userCol, itemCol, rank=5, maxIter=2, spark=None):
        self.userCol = userCol
        self.itemCol = itemCol
        self.rank = rank
        self.maxIter = maxIter
        self.als = ALS(
            rank=self.rank, maxIter=self.maxIter, 
            userCol=self.userCol, itemCol=self.itemCol, 
            seed=666, implicitPrefs=True)
        self.model = None
        self.spark = spark

        
    def fit(self, user_item_sdf):
        self.model = self.als.fit(user_item_sdf)
       
    def transform(self):
        return self.model.recommendForAllUsers(10)

    def save(self, file_name):
        self.model.write().overwrite().save(file_name)

    def load(self, file_name):
        self.model = ALSModel.load(file_name)


