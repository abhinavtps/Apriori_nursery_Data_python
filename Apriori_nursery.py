import pandas as pd
import numpy as np
from collections import namedtuple
from itertools import combinations
from itertools import chain


class TransactionManager(object):
   

    def __init__(self, transactions):
        self.__num_transaction = 0
        self.__items = []
        self.__transaction_index_map = {}

        for transaction in transactions:
            self.add_transaction(transaction)

    def add_transaction(self, transaction):
     
        for item in transaction:
            if item not in self.__transaction_index_map:
                self.__items.append(item)
                self.__transaction_index_map[item] = set()
            self.__transaction_index_map[item].add(self.__num_transaction)
        self.__num_transaction += 1
       
        

    def calc_support(self, items):
      
        # Empty items is supported by all transactions.
        if not items:
            return 1.0

        # Empty transactions supports no items.
        if not self.num_transaction:
            return 0.0

        # Create the transaction index intersection.
        sum_indexes = None
        for item in items:
            indexes = self.__transaction_index_map.get(item)
            if indexes is None:
                # No support for any set that contains a not existing item.
                return 0.0

            if sum_indexes is None:
                # Assign the indexes on the first time.
                sum_indexes = indexes
            else:
                # Calculate the intersection on not the first time.
                sum_indexes = sum_indexes.intersection(indexes)

        # Calculate and return the support.
        return float(len(sum_indexes)) / self.__num_transaction

    def initial_candidates(self):
        return [frozenset([item]) for item in self.items]

    @property
    def num_transaction(self):
        return self.__num_transaction

    @property
    def items(self):
        return sorted(self.__items)

    @staticmethod
    def create(transactions):
        if isinstance(transactions, TransactionManager):
            return transactions
        return TransactionManager(transactions)


# Ignore name errors because these names are namedtuples.
SupportRecord = namedtuple( # pylint: disable=C0103
    'SupportRecord', ('items', 'support'))
RelationRecord = namedtuple( # pylint: disable=C0103
    'RelationRecord', SupportRecord._fields + ('ordered_statistics',))
OrderedStatistic = namedtuple( # pylint: disable=C0103
    'OrderedStatistic', ('items_base', 'items_add', 'confidence', 'lift',))



def create_next_candidates(prev_candidates, length):
    # Sort the items.
    items = sorted(frozenset(chain.from_iterable(prev_candidates)))
    # Create the temporary candidates. These will be filtered below.
    # Creating all combintions of items of length 'length'
    tmp_next_candidates = (frozenset(x) for x in combinations(items, length))
    # Return all the candidates if the length of the next candidates is 2
    # because their subsets are the same as items.
    if length < 3:
        return list(tmp_next_candidates)
    
    #If a 1-item itemset is infrequent then no need to consider that in next itemset
    # Filter candidates that all of their subsets are
    # in the previous candidates.
    next_candidates = [
        candidate for candidate in tmp_next_candidates
        if all(frozenset(x) in prev_candidates for x in combinations(candidate, length - 1))
    ]
    return next_candidates



def gen_support_records(transaction_manager, min_support, **kwargs):
    # Parse arguments.
    max_length = kwargs.get('max_length')
    # Process.
    candidates = transaction_manager.initial_candidates() #returns a list of frozensets containing
    length = 1                                            #one item each
    while candidates:
        relations = set()
        for relation_candidate in candidates:
            support = transaction_manager.calc_support(relation_candidate)
            if support < min_support:
                continue
            candidate_set = frozenset(relation_candidate)
            relations.add(candidate_set)
            yield SupportRecord(candidate_set, support) #yield is used to return the value of a function without destroying the local variables
        length += 1
        if max_length and length > max_length:
            break
        candidates = create_next_candidates(relations, length) # used to create 2-item item set from 1 item frequent itemset and so on


def gen_ordered_statistics(transaction_manager, record):
   
    #input to this function is a object of transactions and all frequent itemsets and their support value as record
    items = record.items
    sorted_items = sorted(items)
    for base_length in range(len(items)):
        for combination_set in combinations(sorted_items, base_length):
            items_base = frozenset(combination_set)
            items_add = frozenset(items.difference(items_base))
            confidence = (
                record.support / transaction_manager.calc_support(items_base))
            lift = confidence / transaction_manager.calc_support(items_add)
            yield OrderedStatistic(frozenset(items_base), frozenset(items_add), confidence, lift)


def filter_ordered_statistics(ordered_statistics, **kwargs):
    min_confidence = kwargs.get('min_confidence', 0.0)
    min_lift = kwargs.get('min_lift', 0.0)

    for ordered_statistic in ordered_statistics:
        if ordered_statistic.confidence < min_confidence:
            continue
        if ordered_statistic.lift < min_lift:
            continue
        yield ordered_statistic


def apriori(transactions, **kwargs):

    # Parse the arguments.
    min_support = kwargs.get('min_support', 0.1)
    min_confidence = kwargs.get('min_confidence', 0.0)
    min_lift = kwargs.get('min_lift', 0.0)
    max_length = kwargs.get('max_length', None)

    # Check arguments.
    if min_support <= 0:
        raise ValueError('minimum support must be > 0')
   
    # Calculate supports.
    transaction_manager = TransactionManager.create(transactions)
    support_records = gen_support_records(transaction_manager, min_support, max_length=max_length)
    
    #support_records will contain list of all those candidate set whose support exceeds minimum support
    
    # Calculate ordered stats.
    for support_record in support_records:
        ordered_statistics = list(filter_ordered_statistics(gen_ordered_statistics(transaction_manager, support_record),min_confidence=min_confidence,
                min_lift=min_lift,))
        if not ordered_statistics:
            continue
        yield RelationRecord(support_record.items, support_record.support, ordered_statistics)



# 1. Reading the data from nursery.csv file
data=pd.read_csv('nursery.csv',names=['parents','has_nurs','form' ,'children','housing','finance','social','health','class_label'])
X=data.iloc[:,:].values
Y=X

# Function to find whether a rule is subset of another
def subset(l1,l2) :
    if(set(l1).issubset(set(l2))):
        return 1
    return 0

# 2. Preprocessing the data     
for i in range(9):
    for j in range(12960) :
        X[j,i]= data.columns[i] +'_'+ X[j,i] 

transactions =[]
for i in range(12960) :
    transactions.append([str(X[i,j]) for j in range(9)])
    result = apriori(transactions,max_length=5,min_support=1,min_confidence=0.95)
M=(list(result))
r=[list(M[i][0]) for i in range(len(M))]
r=[]
for item in M:
    o_s = item[2]
    for x in o_s :
        add=[]
        base=[]
        for l in x[1] :
            add.append(l)
        for l in x[0] :
            base.append(l)
        r.append(base.copy())
        r.append(add.copy())
        r.append(item[1])
        r.append(x[2])
        base.append("--------->")
        base.extend(add)
        #print(base)
# 5. Isolating the rules which have decendant as class label
base=[]
nparray = np.array(r)
nparray=nparray.reshape(-1,4)
base=np.array(base)
add=[]
for item in nparray :
    if (item[1][0][0]=='c' and item[1][0][1]=='l'and len(item[1])==1):
        base= np.append(base,item,axis=0)

base=base.reshape(-1,4)

# 6. Refining set of rules such that rules which are superset of 
#    other rule are not considered
  
add=base.copy()
add=np.c_[ add, np.zeros(len(add)) ] 
for i in range(len(add)):

    for x in range(i+1,len(add)):
        
        if(add[x][4]!=1) and subset(add[i][0],add[x][0]) :
            add[x][4]=1

# 7. Generating final set of rules
base = []
for item in add :
    if item[4]==0 :
        base.append(item)
base=np.array(base)
base=np.delete(base,4,axis=1)
base = np.append(np.array([['Base_item','Class_label','Support','Confidence']]),base, axis=0)
rules=[]
for i in range(1,len(base)) :
    j = ","
    j = j.join(base[i][0])
    j = "{" + j + "}"
    j = j + "----------->" + "{" + base[i][1][0] + "}" 
    rules.append(j)
    
print(rules)        
        
# 8. Plotting graph for each rule

# Plotting the confidence of each rule
import matplotlib.pyplot as plt
fig = plt.figure()
ax= fig.add_axes([0,0,1,1])
Confidence=base[1:len(base),3].tolist()
x=[]
for i in range(1,len(base)) :
    x.append(str(i))
ax.bar(x,Confidence)
plt.ylabel('Confidence')
plt.xlabel('Rules')
plt.show()

# Plotting the support of each item set ,the rule belongs to 
fig1 = plt.figure()
ax= fig1.add_axes([0,0,1,1])
Support=base[1:len(base),2].tolist()
x=[]
for i in range(1,len(base)) :
    x.append(str(i))
ax.bar(x,Support)
plt.ylabel('Support')
plt.xlabel('Rules')
plt.show()   

