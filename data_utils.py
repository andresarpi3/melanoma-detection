import numpy as np
import pandas as pd
import tensorflow as tf

age_normalization_fn = lambda x :  x / 90 if np.isfinite(x) else 0.5
sex_nomalization_fn = lambda x : 1.0 if x == "male" else 0.0

class CsvTransformer:
    cols = ['sex', 'age', 'anatom_head/neck', 'anatom_lower extremity', 'anatom_oral/genital', 'anatom_palms/soles', \
        'anatom_torso', 'anatom_upper extremity', 'target']
    
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        if 'target' not in df.columns:
            df['target'] = -1.0
        
        if 'image_name' not in df.columns:
            df['image_name'] = df['image']
        
        df['age'] = df['age_approx'].apply(age_normalization_fn).values
        df['sex'] = df['sex'].apply(sex_nomalization_fn).values
        df['target'] = df['target'].astype("float64")
        df = pd.get_dummies(df,prefix=['anatom'], columns = ['anatom_site_general_challenge'], drop_first=False)
        df = df.set_index('image_name')
        self.df = df
        self.init_tables()
        
    def init_tables(self):
        # build a lookup table
        self.lookup_tables = {}
        for col in self.cols:
            table = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=tf.constant(list(self.df.index.values)),
                    values=tf.constant(list(self.df[[col]].values.astype("float64").flatten())),
                ),
                default_value=tf.constant(-1.0, dtype=tf.float64),
                name="class_weight"
            )
            
            self.lookup_tables[col] = table

    def get_data_vector(self, image_names):
        vals = []
        for col in self.cols:
            if col == 'target':
                continue
            table = self.lookup_tables[col]
            val = table.lookup(image_names)
            vals.append(val)

        return tf.transpose(tf.stack(vals))

    def get_vector_from_image_name(self, col, image_name):
            
        return self.lookup_tables[col].lookup(image_name)





