#i am kind of FREE
from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Dict, Any, List


class DataCleaner:
    """
    Transform a pandas df while keeping track of the history of transformations to
    allow reverting back to earlier state.
    """
    def __init__(self, df: pd.DataFrame):
        self.current = df
        self.history = [('Initial df', self.current.copy(deep=True))]

    def adjust_dtype(self, types: Dict[str, Any]) -> None:
        #Check if key inputted is a valid key
        col_list = set(self.current.columns)
        for key in types:
            if key not in col_list:
                print(f'The key {key} is not in the data.')
                return
        
        #Convert to datatype inputed
        for key in types: 
            self.current[key] = [types[key](cell) for cell in self.current[key]]
            
        #Add updated dataframe to history list, add action to history list
        self.history.append(('Adjusted dtypes using ' + str(types), self.current.copy(deep=True)))

    def impute_missing(self, columns: List[str]) -> None:
        #Calculate mean for each column inputted
        for col in columns:
            mean = self.current[col].mean()
            #Replace NaN with mean value
            self.current[col].replace(np.nan, mean, inplace=True)
        
        #Add updated dataframe to history list, add action to history list
        self.history.append(('Imputed missing in '+ str(columns), self.current.copy(deep=True)))

    def revert(self, steps_back: int = 1) -> None:
        #Check if steps_back is not higher then available steps
        if steps_back >= len(self.history):
            print('The number of steps you are trying to revert is greater then available steps')
            return

        
        #Revert dataframe
        self.current = self.history[-1*steps_back - 1][1]
        
        #Adjust history list
        for x in range(steps_back):
            self.history.pop()

    def save(self, path: str) -> None:
        #Using pickle class to store data
        #Pickle class will store the datatypes of each column unlike a CSV or Text file
        import pickle
        filepath = path + ".pickle"
        with open(filepath,"wb") as f:
            pickle.dump(transactions_dc.history, f)

    @staticmethod
    def load(path: str) -> DataCleaner:
        #Use pickle to open the file
        import pickle
        filepath = path + ".pickle"
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        #Initialize DataCleaner object
        DC = DataCleaner(data[-1][-1])
        DC.history = data
        return DC

transactions = pd.DataFrame(
    {
        "customer_id": [10, 10, 13, 10, 11, 11, 10],
        "amount": [1.00, 1.31, 20.5, 0.5, 0.2, 0.2, np.nan],
        "timestamp": [
            "2020-10-08 11:32:01",
            "2020-10-08 13:45:00",
            "2020-10-07 05:10:30",
            "2020-10-08 12:30:00",
            "2020-10-07 01:29:33",
            "2020-10-08 13:45:00",
            "2020-10-09 02:05:21",
        ]
    }
)

transactions_dc = DataCleaner(transactions)

print(f"Current dataframe:\n{transactions_dc.current}")

# Current dataframe:
#    customer_id  amount            timestamp
# 0           10    1.00  2020-10-08 11:32:01
# 1           10    1.31  2020-10-08 13:45:00
# 2           13   20.50  2020-10-07 05:10:30
# 3           10    0.50  2020-10-08 12:30:00
# 4           11    0.20  2020-10-07 01:29:33
# 5           11    0.20  2020-10-08 13:45:00
# 6           10     NaN  2020-10-09 02:05:21

print(f"Current dtypes:\n{transactions_dc.current.dtypes}")

# Initial dtypes:
# customer_id      int64
# amount         float64
# timestamp       object
# dtype: object

transactions_dc.adjust_dtype({"timestamp": np.datetime64})

print(f"Changed dtypes to:\n{transactions_dc.current.dtypes}")

# Changed dtypes to:
# customer_id             int64
# amount                float64
# timestamp      datetime64[ns]

transactions_dc.impute_missing(columns=["amount"])

print(f"Imputed missing as overall mean:\n{transactions_dc.current}")

# Imputed missing as mean:
#    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21

print(f"History of changes:\n{transactions_dc.history}")

# ** Any coherent structure with history of changes **
# E.g., here's one possibility

# History of changes:
# [('Initial df',    customer_id  amount            timestamp
# 0           10    1.00  2020-10-08 11:32:01
# 1           10    1.31  2020-10-08 13:45:00
# 2           13   20.50  2020-10-07 05:10:30
# 3           10    0.50  2020-10-08 12:30:00
# 4           11    0.20  2020-10-07 01:29:33
# 5           11    0.20  2020-10-08 13:45:00
# 6           10     NaN  2020-10-09 02:05:21), ("Adjusted dtypes using {'timestamp': <class 'numpy.datetime64'>}",    customer_id  amount           timestamp
# 0           10    1.00 2020-10-08 11:32:01
# 1           10    1.31 2020-10-08 13:45:00
# 2           13   20.50 2020-10-07 05:10:30
# 3           10    0.50 2020-10-08 12:30:00
# 4           11    0.20 2020-10-07 01:29:33
# 5           11    0.20 2020-10-08 13:45:00
# 6           10     NaN 2020-10-09 02:05:21), ("Imputed missing in ['amount']",    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21)]

transactions_dc.save("transactions")
loaded_dc = DataCleaner.load("transactions")
print(f"Loaded DataCleaner current df:\n{loaded_dc.current}")

# Loaded DataCleaner current df:
#    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21

transactions_dc.revert()
print(f"Reverting missing value imputation:\n{transactions_dc.current}")

# Reverting missing value imputation:
#    customer_id  amount           timestamp
# 0           10    1.00 2020-10-08 11:32:01
# 1           10    1.31 2020-10-08 13:45:00
# 2           13   20.50 2020-10-07 05:10:30
# 3           10    0.50 2020-10-08 12:30:00
# 4           11    0.20 2020-10-07 01:29:33
# 5           11    0.20 2020-10-08 13:45:00
# 6           10     NaN 2020-10-09 02:05:21
