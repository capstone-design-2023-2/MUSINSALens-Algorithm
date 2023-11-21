from __future__ import print_function

import os
import pandas as pd
import re

path = 'C:\\Users\\User\\Documents\\mju\\2023 2학기\\캡스톤 디자인\\CBIR'
os.chdir(path)

db_dir = 'database'
db_csv = 'data.csv'

class database:
    def __init__(self):
        self._generate_csv()
        self.data = pd.read_csv(db_csv)
        self.categories = set(self.data['category'])

    def _generate_csv(self):
        #if os.path.exists(db_csv):
        #    return
        with open(db_csv, 'w', encoding='UTF-8') as f:
            f.write("img,category")

            for root, _, files in os.walk(db_dir, topdown=False):
                category = root.split('\\')[-1]

                for name in files:
                    if not name.endswith('.jpg'):
                        continue
                    img = os.path.join(root, name)
                    f.write("\n{},{}".format(img, category))

    def __len__(self):
        return len(self.data)
    
    def get_category(self):
       return self.categories
    
    def get_data(self):
        return self.data
    
if __name__ == "__main__":
    db = database()
    data = db.get_data()
    categories = db.get_category()

    print("DB length: ", len(db))
    print(categories)