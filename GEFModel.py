import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class Data:
    def __init__(self, path=None):
        if path is not None:
            self.read(path)

    def initial(self):
        self.l = np.stack([self.c, self.A.min(axis=0)]).min(axis=0).astype(int).clip(max=0)
        self.u = np.max(self.A, axis=0).astype(int)
        self.zeta = (self.A - self.l).astype(int)

    def write(self, path):
        with open(path, 'w')  as fp:
            fp.write(f"param num_bundles := {self.n_item};\n")
            fp.write("\n")
            fp.write(f"param num_cust := {self.n_buyer};\n")
            fp.write("\n")
            fp.write("param price_reserve :\n")
            a = str(list(range(1, 1+self.n_item))).strip("[]").replace(",", "")
            fp.write(f"   {a} :=\n")
            for b in range(self.n_buyer):
                a = str([i for i in map(int, self.A[b, :].tolist())]).strip("[]").replace(",", "")
                fp.write(f"{b+1} {a}\n")
            fp.write(";\n")
            fp.write("\n")
            fp.write("param:   cost_bundl  marginal :=\n")
            for i in range(self.n_item):
                fp.write(f"{i+1} {self.c[i]} {self.m[i]}\n")
            fp.write(";\n")
            fp.write("param: buyer budget :=\n")
            for i in range(self.n_buyer):
                fp.write(f"  {i} {int(self.budget[i])}\n")
            fp.write(";\n")
            fp.write("param num_partition :=\n")
            fp.write(f"{len(self.items_partition)+1}\n")
            fp.write(";\n")
            fp.write("param: item partition_index cardinality :=\n")
            for i in self.items_partition:
                for item in self.items_partition[i]:
                    fp.write(f"{item} {i} {self.item_card[item]}\n")
            

    def read(self, path):
        with open(path) as fp:
            lines = fp.readlines()
            read_mat = False
            read_cost = False
            wait_partition_num = False
            read_partition = False
            read_budget = False
            item_card = None
            budget = None
            items_partition = None
            for line in lines:
                if 'param' in line:
                    if 'num_bundles'  in line:
                        n_item = int(line.split(':=')[-1].strip(';\n'))
                        cost = np.zeros(n_item)
                        margin = np.zeros(n_item)
                        continue

                    if 'num_cust' in line:
                        n_cust = int(line.split(':=')[-1].strip(';\n'))
                        A = np.zeros((n_cust, n_item))
                        continue
                    
                    if 'price_reserve' in line:
                        # pairing_mat
                        read_mat = True
                        continue
                    
                    if 'cost_bundl' in line:
                        read_cost = True
                        continue

                    if 'num_partition' in line:
                        wait_partition_num = True
                        continue

                    if 'budget' in line:
                        read_budget = True
                        budget = [float('inf')] * n_cust
                        continue
                
                if read_budget:
                    buyer, _budget = list(map(int, line.split()))
                    budget[buyer] = _budget
                    
                    if buyer == n_cust-1:
                        read_budget = False
                    
                    continue

                if wait_partition_num:
                    partition_num = int(line)
                    item_card = [n_item] * n_item
                    items_partition = {i: [] for i in range(partition_num)}
                    wait_partition_num = False
                    read_partition=True
                    continue
                
                if read_partition:
                    if ':=' in line:
                        continue

                    item, part_idx, card = list(map(int, line.split()))
                    items_partition[part_idx].append(item)
                    item_card[item] = card

                    if item == n_item - 1:
                        read_partition = False
                    continue

                if read_mat:
                    if ':=' in line:
                        continue
                    
                    row = line.replace(",", "").split()
                    row_id = int(row[0])
                    row_data = [int(i) for i in row[1:]]
                    A[row_id -1, :] = row_data
                    
                    if row_id == n_cust:
                        read_mat = False
                    continue

                if read_cost:
                    row = line.replace(",", "").split()
                    row_id = int(row[0])
                    row_data = [int(i) for i in row[1:]]
                    cost[row_id-1] = row_data[0]
                    margin[row_id-1] = row_data[1]
                    
                    if row_id == n_item:
                        read_cost = False
                    continue
                
        
        self.n_buyer = n_cust
        self.n_item = n_item
        self.c = cost.astype(int)
        self.m = margin.astype(int)
        self.A = A.astype(int)
        if items_partition is not None:
            self.items_partition = items_partition
        else:
            self.items_partition = {0: self.items}
        
        if item_card is not None:
            self.item_card = item_card
        else:
            self.item_card = [n_item] * n_item
        
        self.budget = budget

        self.initial()
    

    @property
    def items(self):
        return list(range(self.n_item))
    

    @property
    def buyers(self):
        return list(range(self.n_buyer))       
    

    def prefer(self, buyer, i, j):
        # check if buyer prefers item i rather than item j
        if i == j:
            return False
        
        delta = self.A[buyer, i] - self.A[buyer, j]
        
        diff = self.c[i]  - (self.c[j] + delta)
        
        return diff < 0
    

    def follower(self, buyer_b, buyer_a, i, j):
        # check if buyer_a follows buyer_b in preferring item i to j
        # check buyer_b - (k > j) -> buyer_a
        return (self.A[buyer_a, i] - self.A[buyer_a, j]) > (self.A[buyer_b, i] - self.A[buyer_b, j])
    

    def follwer_set(self, buyer_b, buyer_a, j, items=None):
        # return set {k: buyer_b - (k > j) -> buyer_a}
        if items is None:
            items = self.items

        out = []
        for i in items:
            if self.follower(buyer_b, buyer_a, i, j):
                out.append(i)
        return out
    

    def leader_set(self, buyer_b, buyer_a, k, items=None):
        # return set {j: buyer_b - (k > j) -> buyer_a}
        if items is None:
            items = self.items

        out = []
        for i in items:
            if self.follower(buyer_b, buyer_a, k, i):
                out.append(i)
        return out
    

    def superior(self, i, j):
        for b in self.buyers:
            if self.prefer(b, i, j):
                return True
        
        return False
    
    
    def set2_clique(self, i_1, i_2, j, items=None):
        if i_1 == i_2:
            return []
        
        clique = [(i_2, j)] 
        clique += [(i_2, k) for k in self.leader_set(i_1, i_2, j, items)] 
        clique += [(i_1, k) for k in self.follwer_set(i_1, i_2, j, items)]
        
        return clique
    

    def set1_clique(self, i):
        clique = [(i, j) for j in self.items]
        return clique


def build_miblp_model(data, env=None, skip_budget=False, skip_valid=True, skip_opt=True, skip_valid_extra=True):
    m = gp.Model("base", env=env)
    x = m.addVars([(i,j) for i in data.buyers for j in data.items], name='x',  vtype=GRB.BINARY)
    y = m.addVars(data.items, name='y',  vtype=GRB.BINARY)
    p = m.addVars(data.items, name='p', lb=data.l, ub=data.u)
    
    # add constraints

    # valid for all items
    m.addConstrs((x[i, j] <= y[j] for i in data.buyers for j in data.items))
    m.addConstrs((x[i, j] * (data.A[i,j] -p[j]) >= 0 for i in data.buyers for j in data.items))
    m.addConstrs((gp.quicksum(x[i, j] for i in data.buyers) <= data.item_card[j] for j in data.items))

    if not skip_budget and data.budget:
        m.addConstrs((gp.quicksum(p[j] * x[i, j] for j in data.items)<= data.budget[i] for i in data.buyers))

    # valid for partition
    for items in data.items_partition.values():
        m.addConstrs((gp.quicksum(x[i,j] for j in items)<=1 for i in data.buyers))
        
        m.addConstrs((gp.quicksum(x[i,t] * (data.A[i, t] - p[t]) for t in items) >= y[j] * (data.A[i, j] - p[j]) for i in data.buyers for j in items) )

        if not skip_valid:
            m.addConstrs((gp.quicksum(x[i,t] * (data.A[i, t] - p[t]) for t in items) 
                    >= gp.quicksum(x[i_2,t] * (data.A[i, t] - p[t]) for t in items)
                    for i in data.buyers for i_2 in data.buyers if i!=i_2
                    ))
        
        if not skip_opt:
            # opt cut
            constrs_set = m.addConstrs(( x[i,j] + y[k] - 1 <= gp.quicksum(x[r,k] 
                for r in data.buyers if not data.follower(i, r, j, k) and r != i) 
                for i in data.buyers 
                for j in items 
                for k in items 
                if data.prefer(i, k, j)))
            for c in constrs_set:
                constrs_set[c].Lazy=1

        if not skip_valid_extra:
            m.addConstrs((
                #gp.quicksum(x[b, i] for b, i in data.set2_cstr(i_1, i_2, j))<=1
                gp.quicksum(x[i_1, t] * (data.A[i_1, t] - p[t]) for t in items) >= 
                gp.quicksum(x[b, i] * (data.A[i_1, i] - p[i]) for b, i in data.set2_clique(i_1, i_2, j, items)) 
                for i_1 in data.buyers 
                for i_2 in data.buyers
                for j in items
                if i_1 != i_2
                ))
        
    m.setObjective(gp.quicksum( x[i, j] * (p[j]-data.c[j]) for i in data.buyers for j in data.items)
               - gp.quicksum(y[j] * data.m[j] for j in data.items), GRB.MAXIMIZE)
        
    m.update()
    
    var = {'x': x, 'y': y, 'p': p}
    m._vars = m.getVars()   
    
    return m, var


def build_mcc_model(data,
                    skip_budget=False,
                    skip_valid=True,
                    skip_opt=True,
                    skip_valid_extra=True,
    ):
    m = gp.Model("base_mcc")
    x = m.addVars([(i,j) for i in data.buyers for j in data.items], 
                  name='x',  
                  lb = 0,
                  ub = 1,
                  vtype=GRB.BINARY)
    y = m.addVars(data.items, 
                  name='y', 
                  lb = 0,
                  ub = 1,
                  vtype=GRB.BINARY)
    p = m.addVars(data.items, name='p', 
                  lb=data.l, 
                  ub=data.u, 
                  vtype=GRB.CONTINUOUS)
    p_buy = m.addVars([(i,j) for i in data.buyers for j in data.items],
                    #lb=[0 for i in data.buyers for j in data.items],
                    ub=data.A,
                    name='buy')
    
    # add constraints for all items
    m.addConstrs((x[i, j] <= y[j] for i in data.buyers for j in data.items))
    m.addConstrs((data.A[i,j] * x[i,j] >= p_buy[i,j] for i in data.buyers for j in data.items ))
    m.addConstrs((gp.quicksum(x[i, j] for i in data.buyers) <= data.item_card[j] * y[j] for j in data.items))
    m.addConstrs((p_buy[i, j] >= p[j] + x[i, j] * data.u[j] - data.u[j] for i in data.buyers for j in data.items))
    

    if not skip_budget and data.budget:
        m.addConstrs((gp.quicksum(p_buy[i, j] for j in data.items)<= data.budget[i] for i in data.buyers))

    # add constraints for partition
    for items in data.items_partition.values():
        m.addConstrs((gp.quicksum(x[i,j] for j in items)<=1 for i in data.buyers))    
        m.addConstrs((gp.quicksum(x[i,j] * data.A[i, j] - p_buy[i,j] for j in items) \
        + (data.A[i,t] - data.l[t]) * (1-y[t]) 
        >= data.A[i, t] - p[t] for i in data.buyers  for t in items))

        if not skip_valid:
            m.addConstrs((gp.quicksum(x[i,t] * data.A[i, t] - p_buy[i, t] for t in items) 
                >= gp.quicksum(data.A[i, j] * x[i_2, j]- p_buy[i_2, j] for j in items) 
                for i in data.buyers 
                for i_2 in data.buyers  
                if i_2 != i))

        if not skip_opt:
            # opt cut - to be modified
            constrs_set = m.addConstrs(( x[i,j] + y[k] - 1 <= gp.quicksum(x[r,k] 
                for r in data.buyers if not data.follower(i, r, j, k) and r != i) 
                for i in data.buyers 
                for j in items 
                for k in items 
                if data.prefer(i, k, j)))
            for c in constrs_set:
                constrs_set[c].Lazy=1

        if not skip_valid_extra:
            _ = m.addConstrs((
                gp.quicksum(x[i_1, t] * data.A[i_1, t] - p_buy[i_1, t] for t in items) >= 
                gp.quicksum(x[b, i] * data.A[i_1, i] - p_buy[b, i] for b, i in data.set2_clique(i_1, i_2, j, items)) 
                for i_1 in data.buyers 
                for i_2 in data.buyers
                for j in items
                if i_1 != i_2
                ))
    
    m.setObjective(gp.quicksum( p_buy[i,j] - x[i, j] * data.c[j] for i in data.buyers for j in data.items)
               - gp.quicksum(y[j] * data.m[j] for j in data.items), GRB.MAXIMIZE)
    
    var = {'x': x, 'y': y,'buy': p_buy}
    
    m._vars = m.getVars()    
    m.update()
        
    return m, var


def run_test(path, time_limit):
    cols = ['Ins']

    model_selection = ['MINLP', 'MINLP-V', 'MCC', "MCC-V", "MCC-VC"]
    for m in model_selection:
        cols += [f"Primal-{m}", f"Dual-{m}", f"Gap-{m}", f"Time-{m}", f"Node-{m}"]
    df = pd.DataFrame(columns=cols)

    df = df.set_index('Ins')

    del m

    data = Data(path)
    
    for m in model_selection:
        print(f"\n\n------ {m} ------\n\n")
        if m == "MINLP":
            model,  _ = build_miblp_model(data)
        if m == "MINLP-V":
            model, _ = build_miblp_model(data, skip_valid=False)
        if m == "MCC":
            model, _ = build_mcc_model(data)
        if m == "MCC-V":
            model, _ = build_mcc_model(data, skip_valid=False)
        if m == "MCC-VC":
            model, _ = build_mcc_model(data, skip_valid=False, skip_valid_extra=False, skip_opt=False)

        # solve
        model.setParam("TimeLimit", time_limit)
        model.optimize()

        df.loc['test', [f"Primal-{m}", f"Dual-{m}", f"Gap-{m}", f"Time-{m}", f"Node-{m}"]] = [
            model.ObjVal, model.ObjBound, model.MIPGap, model.Runtime, model.NodeCount
            ]
    
    return df
            

def solve(m, time_limit=None, node_limit=None):
    if time_limit is not None:
        m.setParam("TimeLimit", time_limit)
    
    if node_limit is not None:
        m.setParam("NodeLimit", node_limit)
    
    m.optimize()
