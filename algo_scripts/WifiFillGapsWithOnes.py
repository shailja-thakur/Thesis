import pandas as pd
import os
from path import *
path = DATA_PATH + day + USER_ATTRIBUTION_TABLES_PATH + 'location_room_set.csv'

df = pd.read_csv(path)

