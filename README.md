import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import time

start = time.time()
df = pd.read_csv("Groceries_dataset.csv")
end = time.time()
print("time = ",end-start,end="\n")

df['Date'] = pd.to_datetime(df['Date'])
df['itemDescription'] = df['itemDescription'].str.lower()
df = df.drop('Member_number', axis=1)

df_pivot = df.pivot_table(index='Date', columns='itemDescription', aggfunc=lambda x: 1 if len(x)>0 else 0).fillna(0)


def support(x):
    # Support is the fraction of transactions that contain the itemset
    return x.sum() / len(df_pivot)

def confidence(x, y):
    # Confidence is the fraction of transactions that contain the itemset x that also contain the itemset y
    return (x & y).sum() / x.sum()

def lift(x, y):
    # Lift is the ratio of the observed frequency of x and y co-occurring to the expected frequency if they were independent
    return (x & y).sum() / (x.sum() * y.sum())

def conviction(x, y):
    # Conviction is the ratio of the expected frequency that x occurs without y to the observed frequency of x occurring without y
    return (1 - y.sum()) / (1 - confidence(x, y))


def hho(n_horses, n_iter, n_obj, n_items, lower, upper):
    # n_horses: number of horses in the herd
    # n_iter: number of iterations
    # n_obj: number of objectives
    # n_items: number of items
    # lower: lower bound of the itemset size
    # upper: upper bound of the itemset size

 
    print("starting hho function")
    horses = np.random.randint(lower, upper + 1, size=(n_horses, n_items))
    horses = horses.astype(bool)

    
    fitness = np.zeros((n_horses, n_obj))

    
    best_horse = None
    best_fitness = None

    
    for i in range(n_iter):
        print("i = ",i)
        # Loop for each horse
        for j in range(n_horses):
            
            # Evaluate the fitness of the horse
            fitness[j, 0] = support(horses[j])
            fitness[j, 1] = confidence(horses[j], horses[j])
            fitness[j, 2] = lift(horses[j], horses[j])
            fitness[j, 3] = conviction(horses[j], horses[j])

            # Update the best horse if it is better than the current one
            if best_horse is None or dominates(fitness[j], best_fitness):
                best_horse = horses[j].copy()
                best_fitness = fitness[j].copy()

        # Loop for each horse again
        for j in range(n_horses):
            # print("2j = ",j)
            # Generate a random number
            r = np.random.rand()

            # If the random number is less than 0.5, perform exploration
            if r < 0.5:

                
                e = np.random.rand()

                # If the random number is less than 0.5, perform jumping
                if e < 0.5:

                    # Generate a random jump size
                    jump = np.random.randint(-1, 2, size=n_items)

                    # Update the horse by adding the jump size
                    horses[j] = horses[j] + jump

                    # Clip the horse to the bounds
                    horses[j] = np.clip(horses[j], lower, upper)

                    # Convert the horse to boolean
                    horses[j] = horses[j].astype(bool)

                # Else, perform levying
                else:

                    # Generate a random levy index
                    levy = np.random.randint(0, n_horses)

                    # Update the horse by adding the difference between the levy horse and the current horse
                    horses[j] = horses[j] + (horses[levy] ^ horses[j])

                    # Clip the horse to the bounds
                    horses[j] = np.clip(horses[j], lower, upper)

                    # Convert the horse to boolean
                    horses[j] = horses[j].astype(bool)

            # Else, perform exploitation
            else:

                # Update the horse by adding the difference between the best horse and the current horse
                horses[j] = horses[j] + (best_horse ^ horses[j])

                # Clip the horse to the bounds
                horses[j] = np.clip(horses[j], lower, upper)

                # Convert the horse to boolean
                horses[j] = horses[j].astype(bool)

    # Return the best horse and its fitness
    return best_horse, best_fitness

# Define the dominance function
def dominates(x, y):
    # x and y are two vectors of the same length
    # Return True if x dominates y, False otherwise
    # x dominates y if x is better than y in at least one objective and not worse than y in any objective
    print("\n Entering dominates")
    better = False
    for i in range(len(x)):
        if x[i] > y[i]:
            better = True
        elif x[i] < y[i]:
            return False
    return better


#Run the horse herd optimization algorithm
print("\n Before entering hho")
start = time.time()
best_horse, best_fitness = hho(n_horses=50, n_iter=100, n_obj=4, n_items=len(df_pivot.columns), lower=0, upper=1)
end = time.time()
print("Time taken for HHO  = ",end-start)
# Print the best horse and its fitness
print("The best itemset is:", df_pivot.columns[best_horse])
print("The fitness values are:")
print("Support:", best_fitness[0])
print("Confidence:", best_fitness[1])
print("Lift:", best_fitness[2])
print("Conviction:", best_fitness[3])


# Function to predict the next frequent item
def predict_next_frequent_item(df_pivot, input_item, min_support=0.01, min_confidence=0.5):
    # Add the input item to the dataset
    df_pivot[input_item] = 1 if input_item in df_pivot.columns else 0

    # Run Apriori algorithm
    frequent_itemsets = apriori(df_pivot, min_support=min_support, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Find items that are frequent and have a high confidence with the input item
    next_frequent_items = rules[rules['antecedents'].apply(lambda x: input_item in x)]['consequents'].values

    return next_frequent_items


input_item = "whole milk"
next_frequent_items = predict_next_frequent_item(df_pivot, input_item)
print(f"The next frequent items related to '{input_item}' are:", next_frequent_items)
