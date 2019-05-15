from pyspark.sql import Window
import pyspark.sql.functions as F

class Labelizer:
    
    def __init__(self, all_data=None, id_col=None, idx_col=None, spark=None):
        self.all_data = all_data
        self.id_col = id_col
        self.idx_col = idx_col
        self.id_idx = None
        self.spark = spark
        
    def get_id_idx(self):
        self.id_idx = id2idx(self.all_data, self.id_col, self.idx_col)
        return self.id_idx
    
    def save(self, file_name):
        self.id_idx.write.mode("overwrite").parquet(file_name)
    
    def load(self, file_name):
        self.id_idx = self.spark.read.parquet(file_name)
    
    @classmethod
    def create_from_saved(cls, file_name, id_col=None, idx_col=None, spark=None):
        labelizer = cls(all_data=None, id_col=id_col, idx_col=idx_col, spark=spark)
        labelizer.load(file_name)
        return labelizer
    
def id2idx(id_sdf, id_col, idx_col):
    id_window = Window().orderBy(id_col)
    return id_sdf.withColumn(idx_col, F.rank().over(id_window))