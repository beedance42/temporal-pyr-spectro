"""
Run this to quickly create a .txt file with a list of dates, 
in yymmdd format, for running multiple parallel jobs of 1-day pyramids.
"""
from datetime import date
from datetime import timedelta
import sys

if len(sys.argv) != 4:
    print("Command line arguments should be:")
    print("1) dataset name")
    print("2) begin date (YYYY-MM-DD)")
    print("3) number of days total to list in the file")
    exit()

pyr_title = sys.argv[1]
filename = "cluster_dates_" + pyr_title + ".txt"
f = open(filename, "a")


begin = sys.argv[2]
num_days = int(sys.argv[3])

year, month, day = begin.split("-")

begin_date = date(int(year),int(month),int(day))

write_date = begin_date
for i in range(num_days):
    y = write_date.year
    m = write_date.month
    d = write_date.day
    date_string = str(y)[-2:] + str(m).zfill(2) + str(d).zfill(2) + "\n"
    f.write(date_string)
    write_date = write_date + timedelta(days=1)
    
f.close()
