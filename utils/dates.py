from datetime import date
import numpy as np
def int2date(argdate: int):
    year = int(argdate / 10000)
    month = int((argdate % 10000) / 100)
    day = int(argdate % 100)
    return date(year, month, day)


def int2dates(int_dates):
    return [int2date(d) for d in int_dates]
    
