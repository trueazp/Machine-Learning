# IMPORT LIBRARY
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# IMPORT DATASET
# SOURCE : https://www.kaggle.com/shazadudwadia/supermarket
df = pd.read_csv("D:\Kuliah\Semester_5\Machine Learning\datasets\GroceryStoreDataSet.csv", names=[
                 'products'], sep=',')
df.head()
df.shape

# KONVERSI DATA MENJADI LIST
data = list(df["products"].apply(lambda x: x.split(",")))
data

# KONVERSI TRUE / FALSE -> 1 / 0
# Menggunakan TransactionEncoder, kita konversi daftar menjadi daftar Boolean terenkode.
# Produk yang dibeli atau tidak dibeli pelanggan selama berbelanja sekarang akan diwakili oleh nilai 1 dan 0.
a = TransactionEncoder()
a_data = a.fit(data).transform(data)
df = pd.DataFrame(a_data, columns=a.columns_)
df = df.replace(False, 0)
df = df.replace(True, 1)
df

# PEMODELAN APRIORI
# Kita dapat mengubah semua parameter dalam Model Apriori dalam paket mlxtend.
# Saya akan mencoba menggunakan parameter dukungan minimum untuk pemodelan ini.
# Untuk ini, saya menetapkan nilai min_support dengan nilai ambang 10% dan mencetaknya di layar juga.
df = apriori(df, min_support=0.1, use_colnames=True, verbose=1)
df

# SHOW SUPPORT, CONFIDENCE, LIFT
# Saya memilih nilai Confidence minimum sebanyak 60%.
# Dengan kata lain, ketika produk X dibeli, kita dapat mengatakan bahwa pembelian produk Y adalah 60% atau lebih.
df_ar = association_rules(df, metric="confidence", min_threshold=0.6)
df_ar

# MOHON AGAR PROGRAM DIJALANKAN MENGGUNAKAN jupyter-notebook ataupun google-collab
