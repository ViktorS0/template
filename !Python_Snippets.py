# main libraries
import numpy as np
import pandas as pd
from pathlib import Path

import psycopg2

from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# load function from selected location
import sys
sys.path.insert(0, path2func)
# reload myfunc each time a function from this file is called
%load_ext autoreload
%autoreload 2
path2func = Path("/Users/viktor.sokolov/Documents")
import sys
sys.path.insert(0, path2func)
from myfunc import *


# set options to display all columns
from IPython.display import display
pd.options.display.max_columns = None
# set options to display numbers in not scientific way - 3 digits after the dot
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True)
# dispalay full (non-truncated) dataframe
pd.set_option('display.max_colwidth', None)

# set Jupyter Notebook to diplay all results not just the last one
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# ML libraries
from sklearn import datasets

# Set ups
# plot stiles
sns.set()
plt.style.use('ggplot')
plt.style.use('fivethirtyeight')

# random seed
np.random.seed(42)


# connect to db params 
dbname_eatlip = 'etleap-remix'
dbname_snowplow = 'sp'
host = 'remix-sp.cyw4paki3z7b.eu-central-1.redshift.amazonaws.com'
port = '5439'
user = 'viktor_sokolov'#Your crendetials 
password = 'B4aqP73NPqwBqoiNtTXv' #Your crendetials 

conn_etleap = psycopg2.connect(dbname=dbname_eatlip, host=host, port=port, user=user, password=password)
conn_sp = psycopg2.connect(dbname=dbname_snowplow, host=host, port=port, user=user, password=password)

def run_query(sql, connection, dtype=None):
    """ Execute SQL query
    """ 
    return pd.read_sql_query(sql, connection, dtype=dtype)


query = """
select max(id) from ds.{table}
""".format(table=table)

df = run_query(query, conn_etleap)


# Load data
path2data = Path("/Users/viktor.sokolov/Documents")
path2write = Path("/Users/viktor.sokolov/Documents")
filename = '.csv'
df= pd.read_csv(path2data / filename, nrows=5, header=None, encoding = "ISO-8859-1",  comment='#', na_values=['Nothing']
                , dtype={'col_name': str, 'col_name' : int, 'col_name' : float}
                , parse_dates=['date'])

print('shape:', df.shape)
df[:2]

## read data many files
import glob 
allFiles = glob.glob(path2file_bim + "/*.txt")

df = pd.DataFrame()
list_ = []
### read all files in dir
for file_ in allFiles:
    frame = pd.read_csv(file_, sep='\t', encoding = "ISO-8859-1", usecols=col1)
    list_.append(frame)
    
# merge data in one file    
df = pd.concat(list_, ignore_index=True)

# check the data
df.head()
df.info()
df.describe()

# write data
df.to_csv(path2write / 'file_name.csv', index=False)

# calculate process time 
import time
start_time = time.process_time()

from datetime import timedelta
print(timedelta(seconds=time.process_time()-start_time))

# merge df
df1 = pd.merge(df1, df2, on='colum_name', how='left')

# update a column by reference 
dict1 = {key: value for key, value in zip(df1['key'], df1['value'])}
dict1 = dict(zip(df1['key'], df1['value']))
dict1 = {'key1':'val1', 'key2':'val2'}
df2['column_name'] = df2['key'].map(dict1) # map column key by dict1.keys() and update/ crete column df2['column_name'] with valies in dict1.values()

# aggregate 
df_agg = df.groupby(['colum_name1', 'colum_name2'], as_index=False)['column_to_sum'].agg('sum')
df_agg = df.groupby(['colum_name1', 'colum_name2'], as_index=False).agg(name=('column', 'aggfunc'))
## pass arguments 
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

slip_level_reg.groupby(['colum_name1', 'colum_name2'], as_index=False).agg(name1=('column', percentile(40)), name2=('column', percentile(70)))
df.groupby(['Group1', 'Group2'], as_index=False).agg(name1=('column1', lambda x: np.percentile(x, 40)),
                                                              name2=('column2', lambda x: np.percentile(x, 70))

# by time
df.groupby([pd.Grouper(key='datetime', freq='H')]).agg(unique_items=('item_code', 'nunique'),total_quantity=('quantity','sum'))
df.reset_index(level=0, inplace=True)
                                                                        
#-------------------------------------------------------------------------------------
# MY FUNCTIONS
#-------------------------------------------------------------------------------------
# all combination                  
def partitions(set_):
	"""
	A generator functions that generates all possible combinations from the values in the list
	Takes a list and creates all possible combinations from the values in the list
	
	Parameters:
    set_ - a list	
	"""
    if not set_:
        yield []
        return
    for i in range(2**len(set_)//2):
        parts = [set(), set()]
        for item in set_:
            parts[i&1].add(item)
            i >>= 1
        for b in partitions(parts[1]):
            yield [parts[0]]+b

#-------------------------------------------------------------------------------------
# Calc the number of batches in tuple(x) by specified batch size (b_size)
def make_batches(x, b_size=1):
    """
    calc the number of batchies in tuple x by specified batch size (b_size)
    
    IN:
        x - takes a tuple, list or pd.series to be split in batches
        b_size=10 - specify the size of the batches: 1, 2**3, 10...
    OUT:
        n_batches - the number of batches
    Example:
		t1 = (1,2,3,4,5)
		b_size=2
		n_batchies = make_batches(x=t1, b_size=b_size)
		print('Number of batches: ', n_batchies)
		print('-------------------------------------------------------------')

		#Iterate over batches
		for i in range(n_batchies):
			s=i*b_size; e=(i+1)*b_size
			print('extracting batch: ', i, 'from: ', s, 'to: ', e)
			print(t1[s:e])
			print('-------------------------------------------------------------')
			
    
    """
    n_batches =  (len(x) + b_size - 1) // b_size
    
    return n_batches

#-------------------------------------------------------------------------------------	
# This is a helper function that will fetch all of the available 
# partitions for you to use for your brute force algorithm.
def get_partitions(set_):
    for partition in partitions(set_):
        yield [list(elt) for elt in partition]

### Uncomment the following code  and run this file
### to see what get_partitions does if you want to visualize it:

#for item in (get_partitions(['a','b','c','d'])):
#     print(item)

#-------------------------------------------------------------------------------------	
# Detect outliers - IQR outlier detection
def detect_outliers(x, sensitivity = 1.5, transformation="no"):
    ''' IQR outlier detection 
    IN: 
        - x: a numeric pandas series 
        - sensitivity: numeric (1.5:2.0) - identifies sensitiviy of outlier detection. 
        The biger the number less sensitive the detection.
        - transformation: alowed values('no', 'log', 'sqrt') - how the input to be transformed
    OUT:
        - pd.series identifying outliers ("no", "high_outlier", "low_outlier") indexed as the input
         
    '''
    assert x.dtype=='float64'
    assert (sensitivity >= 1.5) & (sensitivity <= 2.0)
    assert (transformation=='no') | (transformation=='log') | (transformation=='sqrt')
    
    x.rename('values', inplace = True)
    # transform data
    if transformation=='no':
        x_transfomed  = pd.DataFrame(x.copy())
    elif transformation=='log':
        x_transfomed = pd.DataFrame(np.log(x+x[x>0].min()/2))
    elif transformation=='sqrt':
        x_transfomed  = pd.DataFrame(x**0.5)
    else:
        raise ValueError("transformation can only be one of ('no', 'log', 'sqrt')")

    # Identify outlier thresholds
    xq = x_transfomed.quantile(([.25, .75]))
    IQR = xq.loc[0.75,:]-xq.loc[0.25,:]
    low_threshold = xq.loc[0.25,:] - (sensitivity * IQR)
    high_threshold = xq.loc[0.75,:] + (sensitivity * IQR)

    # mark outliers
    x_transfomed['outlier'] = 'no'
    x_transfomed.loc[x_transfomed['values']<low_threshold[0], 'outlier']='low_outlier'
    x_transfomed.loc[x_transfomed['values']>high_threshold[0], 'outlier']='high_outlier'
    
    return(x_transfomed['outlier'])

#-------------------------------------------------------------------------------------
# OPIMIZATION
#-------------------------------------------------------------------------------------
# cows to transport: cow1 weights 4...  
cows = {cow1:4, cow2:2, cow3:6, cow4:5}

# optimize the transport of cows so that it takes min number of trips when one trip can only take maximum weight of 10
def brute_force_cow_transport(cows,limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # find the weight for all combinations
    all_comb = []
    all_w = []
    for item in (get_partitions(cows)):
        all_comb.append(item)
        w0 = []
        for i in range(len(item)):
            # print(i, w)
            w1 = 0.0
            for e in item[i]:
                w1 += cows.get(e)
            w0.append(w1)
        all_w.append(w0)
    
    # find the combinations that fir within the weight limit
    c0=0
    useful_comb = []
    useful_w = []
    for w in all_w:
        if all(x <= limit for x in w):
            useful_comb.append(all_comb[c0])
            useful_w.append(w)
        c0+=1
    
    # select the min trips
    useful_comb = sorted(useful_comb, key = len)
    
    return useful_comb[0]

 
# not always  optimization but fast
def greedy_cow_transport(cows,limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # TODO:  Your code here    
    itemsCopy = sorted(cows.items(), key = lambda v: v[1], reverse = True)
    
    all_results = []
    while len(itemsCopy) > 0: 
        result = []
        Tval = 0
        to_drop = []
        for item in itemsCopy:            
            if Tval + item[1] <= limit:
                result.append(item[0])
                Tval += item[1]
                to_drop.append(item)
        
        for i in to_drop:
            itemsCopy.remove(i)

        all_results.append(result)
        
    return all_results

#-------------------------------------------------------------------------------------
# ALL FILE IN DIR
#-------------------------------------------------------------------------------------
# find all files in a directory
def find_files(path = None, pattern1 = None, pattern2 = None):
    '''
    Return lists with all files and directories under the path 
    IN: 
        path: a string path name e.i. "C/user/..."
        pattern1, pattern2: a string with a pattern like '.csv' that will be used find files fitting the pattern/s
    OUT: two lists = files_paths; files_names
        files_paths: list of all directories 
        files_names: list of all file names
    '''
    if path == None:
        raise ValueError('Please provide a directory - i.e "C/user/..."')
        
    import os
    
    files_paths = []
    files_names = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        if pattern1 != None and pattern2 != None:
            for file in f:
            # select criteria to filter the files by: 'txt' files and 'New' in the name           
                if pattern1 and pattern2 in file:
                    files_paths.append(os.path.join(r, file))
                    files_names.append(file)
            
        if pattern1 != None and pattern2 == None:
            for file in f:
                if pattern1 in file:
                    files_paths.append(os.path.join(r, file))
                    files_names.append(file)
        
        if pattern1 == None and pattern2 == None:
            for file in f:
                    files_paths.append(os.path.join(r, file))
                    files_names.append(file)

    return([files_paths, files_names])




