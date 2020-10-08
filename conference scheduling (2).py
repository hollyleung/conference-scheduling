import numpy as np
import itertools
import gurobipy as gp
from gurobipy import GRB

def phase1(filename,n):
    """
    load data (attender profiles) as np 2d array
    """
    profiles = np.loadtxt(open(filename,"rb"), delimiter=',', skiprows=1).astype(int)                   #skip the first row because it's not actual data
    profiles = np.delete(profiles, 0, axis=1)                                                           #delete the first column because it's not data
    no_talks = profiles.shape[1]                                                                        #no. of talks
    talks = np.linspace(0, no_talks-1, num=no_talks).astype(int)                                        #array of indexs for the talks
    e = list(itertools.combinations(talks, n))                                                          #contains all the n-tuples 

    """
    computing ce (ce being the coefficient for missing a talk, as named in the paper)
    """
    c = []
    for combi in e:                                     #calculate ce for each combination of n talks
        relevant_profiles = profiles[:, list(combi)]    #pick out the relevant columns for this current combination
        ce = relevant_profiles.sum(axis=1)-1            #talks that will be missed by each individual in an array (ce for this e)
        ce = np.maximum(ce,0)                           #elementwise max with 0 to get rid of negative values (see the paper for explanation)
        c.append(ce.sum())                              #sum up missed talks for each person and store in ce
    c = np.array(c)                                     #turn into numpy array for optimisation

    """
    setting up sparse matrix to represent constraints
    """
    A = np.zeros((no_talks,len(e))).astype(int) #initialise matrix A (constraints), the number of constraints will be the same as the number of talks
    for i in range(A.shape[0]):                 #loop over each element of A, i is the intex of the talk
        for j in range(A.shape[1]):             #j is the index of the combination in e
            if i in e[j]:                       #change the corresponding element in A to 1 if the i-th talk is in the j-th combination
                A[i][j] = 1

    """
    setting up optimisation model
    """
    m = gp.Model("phase1")
    x = m.addMVar(shape=len(e), vtype=GRB.BINARY, name="x") #adding binary vector variable "x" containing each of xe in a vector
    m.setObjective(c @ x, GRB.MINIMIZE)                     #setting objective function as the inner product of c and x
    rhs_vector= np.ones(no_talks)                           #right hand side of the equation (all equal to 1)
    m.addConstr(A @ x == rhs_vector, name = "c")            #add Ax = 1 constraint (equation 2 in paper)

    """
    running the model and interpreting the results
    """
    m.optimize()                            #optimises the model
    print("phase 1 missed: %g" %m.objVal)   #output the number of missed preferences
    e = np.array(e)
    chosen_combi = e[x.X.astype(bool)]
    print("combinations are:")
    print(chosen_combi)                     #output the chosen combinations (x.X is the value of the optimised x. here I have made it into boolean values and pick out entries of e where the optimised value is 1)
    
    return chosen_combi,profiles

def phase2(combi,k,profiles):
    combi_index = range(len(combi))
    k_blocks_combinations = list(itertools.combinations(combi,k))   #generate sets of k combinations
    k_blocks_combinations_index = list(itertools.combinations(combi_index,k))
    no_combi = len(combi)                                           #no. of available parallel talks combinations
    """
    computing w
    """
    print("computing w")
    wb = []                                  
    block_arrangement = []                                          #records the arrangement for the least hops with each k_set
    for k_set in k_blocks_combinations:
        hops_for_each = []
        k_blocks = generate_k_block_permutations(k_set)             #generates all possible permutations within the set i.e. permuting e1,e2... and within each e
        print("computing hops")
        for block in k_blocks:                                      #go through each k_blocks and compute the minimum number of hops possible
            hops_for_each.append(compute_hops(block,profiles))
        
        wb.append(min(hops_for_each))                               #the weight wb will be the value of minimum hops for each block
        block_arrangement.append(k_blocks[hops_for_each.index(min(hops_for_each))]) #store the optimal arrangement given the set
    print("compute w success")
    """
    setting up sparse matrix to represent constraints (similar to phase 1)
    """
    A = np.zeros((no_combi,len(wb))).astype(int)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):             
            if i in k_blocks_combinations_index[j]:                       
                A[i][j] = 1
    
    """
    setting up optimisation model (similar to phase 1)
    """
    m = gp.Model("phase 2")
    y = m.addMVar(shape=len(k_blocks_combinations), vtype=GRB.BINARY, name='y')
    m.setObjective(np.array(wb) @ y, GRB.MINIMIZE)
    rhs_vector = np.ones(no_combi)
    m.addConstr(A @ y == rhs_vector,name = 'c')
    
    """
    running the model and interpret results (similar to phase 1)
    """
    m.optimize()
    print("phase 2 hops: %g" %m.objVal)
    block_arrangement = np.array(block_arrangement)
    chosen_arrangements = block_arrangement[y.X.astype(bool)]
    print("chosen block arrangements are:")
    print(chosen_arrangements)
    
    return chosen_arrangements

def generate_k_block_permutations(k_set):
    """ 
    generates all the possible permutation of k-block given a set of k combinations
    """
    permuted = list(itertools.permutations(k_set,len(k_set)))                   #get the set of all permutations of e1,e2...ek
    all_k_blocks = []
    for permu in permuted:                                                      #go through each permutation of e
        e_permutations = []
        
        for e in permu:                                                         #compute permutations within each e i.e. swapping between rooms
            e_permutations.append(list(itertools.permutations(e,len(e))))       #add the set of permutations within e to the list
            
        all_k_blocks = all_k_blocks + list(itertools.product(*e_permutations))  #generate the set of feasible k_blocks by taking the cartesean product of each possible permutation within e
                                                                                #note that this uses + operator instead of .append() so that all k_blocks are on the same level
    return all_k_blocks


def compute_hops(block,profiles):
    """
    computes the total number of hops associated with a certain block structure
    """
    total_hops = 0                                                      #total number of hops across everyone
    
    #loop through each person to calculate hops
    for person in profiles:
        interested_talks = np.zeros(np.array(block).shape)              #generate a grid of the same size of block

        for i in range(interested_talks.shape[0]):
            interested_talks[i] = list(map(lambda x: person[x],block[i]))   #put the value of the person's pereference in slots where this person has interest in, so it's 1 if the person is interested in the talk in this slot
        
        options = list(map(lambda x: list(np.where(x==1)[0]),interested_talks))        #list of the options for each time slots
        options = list(map(lambda x: x if x else [-1],options))                     #change all empty sets to -1 to handle empty sets
        possible_routes = list(itertools.product(*options))                 #tuples of all the possible routes to take for this person
        
        #calculate the number of hops for each route, the person will take the route with minimum hops
        hops = []
        for route in possible_routes:               #loop for all routes to calculate hops needed
            previous = route[0]
            route_hops = 0
            for slot in range(len(route)-1):        #loop through the route, add 1 to hops if the person has to hop (i.e. route[i] changes)
                now = route[slot+1]
                if (previous != now) and (now != -1) and (previous != -1): #doesn't add if now or previous is -1 because -1 indicates no preference
                    route_hops = route_hops + 1
                previous = route[slot+1]
            
            hops.append(route_hops)                 #add to the list of number of hops
        
        total_hops = total_hops + min(hops)         #min(hops) is the number of hops needed for this person

    return total_hops


def phase3(filename,n,k,no_talks,arrangements):
    """
    load availability profiles
    """
    profiles = np.loadtxt(open(filename,"rb"), delimiter=',', skiprows=1).astype(int)
    profiles = np.delete(profiles, 0, axis=1)
    no_slots = int(no_talks/(n*k))                          #number of slots (number of slots divided by number of talks in each block (n*k))
    slots = range(no_slots)                                 #list from 1 to no_slot, representing each slot
    pairs = list(itertools.product(arrangements,slots))     #list of every pairing of blocks and timeslots. note that the order of product matters. this has been designed such that the
                                                            #pairs list corresponds to the flattened matrix later
    
    """
    compute weights ub,t'
    """
    u = np.zeros((no_slots,no_slots))                                                   #u is a matrix of weights, each row corresponds to a block, each column corresponds to a block timeslot
    for b in range(no_slots):                                                           #loop through every b and t_
        for t_prime in range(no_slots):
            u[b][t_prime] = compute_violations(profiles,arrangements[b],t_prime,k,n)    #compute number of presenter violations if block b is placed at timeslot t_prime
    
    weight = u.flatten()                #make weighting vector from the matrix
    
    """
    set up constraint matrices
    """
    A = np.zeros((no_slots,len(weight))).astype(int)
    B = np.zeros((no_slots,len(weight))).astype(int)
    for t in range(no_slots):                           #constraints according to equations 8 and 9 in paper: summing all b for t is the same as setting a column in a matrix to 1 and flattening it
        temp_matrix = np.zeros((no_slots,no_slots))
        temp_matrix[:,t] = np.ones(no_slots)
        A[t,:] = temp_matrix.flatten()
    for b in range(no_slots):                           #constraints according to equations 8 and 9 in paper: summing all t for b is the same as setting a row in a matrix to 1 and flattening it (text me if really stuck. this bit is a bit hard to explain)
        temp_matrix = np.zeros((no_slots,no_slots))
        temp_matrix[:,b] = np.ones(no_slots)
        B[b,:] = temp_matrix.flatten()
    
    """
    setting up optimisation model (see phase 1)
    """
    m = gp.Model("phase 3")
    z = m.addMVar(shape=len(weight), vtype=GRB.BINARY, name='z')
    m.setObjective(weight @ z, GRB.MINIMIZE)
    rhs_vector = np.ones(no_slots)
    m.addConstr(A @ z == rhs_vector,name = "c2")
    m.addConstr(B @ z == rhs_vector,name = "c1")
    
    """
    run the model and intepret the result (see phase 1)
    """
    m.optimize()
    print("phase 3 violations: %g" %m.objVal)
    final_arrangements = np.array(pairs)[z.X.astype(bool)]  #outputs timeslot - block pairs for the optimal timetable
    print(final_arrangements)
    return final_arrangements
    
    
def compute_violations(profiles,block,t_prime,k,n):
    total = 0
    for t in range(k):
        total = total + sum(profiles[np.array(block[t])][:,t_prime*k+t])
    violations = n*k - total
    return violations

if __name__ == "__main__":
    [combi,profiles] = phase1("MAPSP2015_instance.csv",3)
    arr = phase2(combi,2,profiles)
    final = phase3("availability.csv",3,2,90,arr)