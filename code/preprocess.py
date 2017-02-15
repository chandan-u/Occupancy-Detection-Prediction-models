

import csv

from datetime import datetime

import time 




f = open('./datasets/datatest2.txt', 'rb')


new_f = open('./datasets/processed_training.csv', "ab")
data = csv.reader(f)
new_data = csv.writer(new_f, delimiter=',' , quoting=csv.QUOTE_NONE)

first = 0
for row in data:


    # skip the first row 
    if first == 0:
        first = 1
        continue

    id = row[0]
    datetime_str  = row[1]

    # "1","2015-02-04 17:51:00",23.18,27.272,426,721.25,0.00479298817650529,1

    dt_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    hour =  dt_obj.strftime("%d")
    month = dt_obj.strftime("%m")

    new_row = [month, hour, row[2], row[3], row[4], row[5], row[6], row[7]]
    new_data.writerow(new_row)
 

 
f.close()
new_f.close()
