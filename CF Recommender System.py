# Program Name: NSGA-base CF Recommender System.py
# Author: Torkashvand 

import sys
import os

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import math
pop_size = 20
max_gen = 921

new = pd.read_csv('u.data', sep='\t',names= ['userID','movieID','rating','timestamp'])
new = new.pivot(index='userID',columns='movieID',values='rating')
# load training data, set new2 as our original training dataframe 
new2 = pd.read_csv('ua.base', sep='\t',names= ['userID','movieID','rating','timestamp'])
new2 = new2.pivot(index='userID',columns='movieID',values='rating')
new3 = pd.read_csv('ua.test', sep='\t',names= ['userID','movieID','rating','timestamp'])
new3 = new3.pivot(index='userID',columns='movieID',values='rating')

        max_k = len(users) - 1
        min_k = 1
        if self.k < min_k or self.k > max_k:
            self.quit("k value for k nearest neighbors is not valid, it should be inside [" + str(min_k) + ", " +  str(max_k) +"]")
        """
    
    def Similarity(self, user1, user2):
        result = 0.0
        user1_data = self.uu_dataset[user1]
        user2_data = self.uu_dataset[user2]

        rx_avg = self.user_average_rating(user1_data)
        ry_avg = self.user_average_rating(user2_data)
        sxy = self.common_items(user1_data, user2_data)

        top_result = 0.0
        bottom_left_result = 0.0
        bottom_right_result = 0.0
        for item in sxy:
            rxs = user1_data[item]
            rys = user2_data[item]
            top_result += (rxs - rx_avg)*(rys - ry_avg)
            bottom_left_result += pow((rxs - rx_avg), 2)
            bottom_right_result += pow((rys - ry_avg), 2)
        bottom_left_result = math.sqrt(bottom_left_result)
        bottom_right_result = math.sqrt(bottom_right_result)

        result = top_result/(bottom_left_result * bottom_right_result)
        return result

    def user_average_rating(self, user_data):
        avg_rating = 0.0
        size = len(user_data)
        for (movie, rating) in user_data.items():
            avg_rating += float(rating)
        avg_rating /= size * 1.0
        return avg_rating

    def common_items(self, user1_data, user2_data):
        result = []
        ht = {}
        for (movie, rating) in user1_data.items():
            ht.setdefault(movie, 0)
            ht[movie] += 1
        for (movie, rating) in user2_data.items():
            ht.setdefault(movie, 0)
            ht[movie] += 1
        for (k, v) in ht.items():
            if v == 2:
                result.append(k)
        return result

    
    def k_nearest_neighbors(self, user, k):
        neighbors = []
        result = []
        for (user_id, data) in self.uu_dataset.items():
            if user_id == user:
                continue
            upc = self.similarity(user, user_id)
            # upc = round(upc, 11)
            neighbors.append([user_id, upc])
            # neighbors_ht.setdefault(user_id, upc)   # assume there are not duplicate user_id
        # sorted_neighbors_ht = sorted(neighbors_ht.iteritems(), key=lambda neighbors_ht : neighbors_ht[1], reverse=True)  
        sorted_neighbors = sorted(neighbors, key=lambda neighbors: (neighbors[1], neighbors[0]), reverse=True)   # - for desc sort

        # testitems = [('a', 3), ('o', 5), ('g', 6), ('c', 1), ('b', 1)]
        # sorted_testitems = sorted(testitems, key=lambda testitems: (-testitems[1], testitems[0]))  # - for desc sort

        for i in range(k):
            if i >= len(sorted_neighbors):
                break
            result.append(sorted_neighbors[i])
        return result

    
    def predict(self, user, item, k_nearest_neighbors):
        valid_neighbors = self.check_neighbors_validattion(item, k_nearest_neighbors)
        if not len(valid_neighbors):
            return 0.0
        top_result = 0.0
        bottom_result = 0.0
        for neighbor in valid_neighbors:
            neighbor_id = neighbor[0]
            neighbor_similarity = neighbor[1]   # Wi1
            rating = self.uu_dataset[neighbor_id][item] # rating i,item
            top_result += neighbor_similarity * rating
            bottom_result += neighbor_similarity
        result = top_result/bottom_result
        return result

    def check_neighbors_validattion(self, item, k_nearest_neighbors):
        result = []
        for neighbor in k_nearest_neighbors:
            neighbor_id = neighbor[0]
            # print item
            if item in self.uu_dataset[neighbor_id].keys():
                result.append(neighbor)
        return result

    
    def load_data(self, input_file_name):
        """
        load data and return three outputs for extention purpose
        only one output is enough in practice (uu_dataset)

        """
        input_file = open(input_file_name, 'rU')
        dataset = []
        uu_dataset = {}
        ii_dataset = {}
        for line in input_file:
            row = str(line)
            row = row.split("\t")
            row[2] = row[2][:-1]
            dataset.append(row)

            """
            user-user dataset: [0: Movie Name  1: Rating]

            """
            uu_dataset.setdefault(row[0], {})
            uu_dataset[row[0]].setdefault(row[2], float(row[1]))
            # uu_dataset[row[0]].append([row[2],row[1]])

            """
            item-item dataset: [0: user id  1: Rating]

            """
            ii_dataset.setdefault(row[2], {})
            ii_dataset[row[2]].setdefault(row[0], float(row[1]))
            # ii_dataset[row[2]].append([row[0], row[1]])
        return dataset, uu_dataset, ii_dataset

    def display(self, k_nearest_neighbors, prediction):
        for neighbor in k_nearest_neighbors:
            print neighbor[0], neighbor[1]
        print "\n"
        print prediction

    def quit(self, err_desc):
        tips = "\n" + "TIPS: " + "\n"   \
                + 

    # publish
    input_file_name = sys.argv[1]   # ratings-dataset.tsv
    user_id = sys.argv[2]   # user name
    movie = sys.argv[3]     # movie name
    k = int(sys.argv[4])    # k neighbors

    # test
    # input_file_name = "ratings-dataset.tsv"
    # user_id = "Kluver"
    # movie = 'The Fugitive'
    # k = 10

    cf = Collaborate_Filter(input_file_name, user_id, movie, k)
    cf.initialize()
        

    # cf.similarity(user_id, user_id)
    # cf. similarity ("Flesh", "Nathan_Studanski")

    k_nearest_neighbors = cf.k_nearest_neighbors(user_id, k)
    # cf.k_nearest_neighbors("Flesh", 2)

    prediction = cf.predict(user_id, movie, k_nearest_neighbors)
    cf.display(k_nearest_neighbors, prediction)

    

ind=recom.intersection(u2.index)

        for i in range(0,len(ind)):

          us1.append(u1.loc[ind[i]])
          us2.append(u2.loc[ind[i]])

        if len(ind)>2:# and len(dif)>70 :

          num=sum([a*b for a,b in zip(us1,us2)])

          std1=np.sqrt(sum([a*b for a,b in zip(us1,us1)]))
          std2=np.sqrt(sum([a*b for a,b in zip(us2,us2)]))

          sim=(num/(std1*std2))+1

        else :sim=0

        Plist.append(sim)
      return Plist
#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

def crossover(a,b):
    r=random.random()
    if r>0.5:
        return mutation((a+b)/2)
    else:
        return mutation((a-b)/2)

#Function to carry out the mutation operator
def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob <1:
        solution = min_x+(max_x-min_x)*random.random()
    return solution

solution=[min_x+(max_x-min_x)*random.random() for i in range(0,pop_size)]
gen_no=0
while(gen_no<max_gen):
    function1_values = [function1(solution[i])for i in range(0,pop_size)]
    function2_values = [function2(solution[i])for i in range(0,pop_size)]
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    print("The best front for Generation number ",gen_no, " is")
    for valuez in non_dominated_sorted_solution[0]:
        print(round(solution[valuez],3),end=" ")
    print("\n")
    crowding_distance_values=[]
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
    solution2 = solution[:]
    #Generating offsprings
    while(len(solution2)!=2*pop_size):
        a1 = random.randint(0,pop_size-1)
        b1 = random.randint(0,pop_size-1)
        solution2.append(crossover(solution[a1],solution[b1]))
    function1_values2 = [function1(solution2[i])for i in range(0,2*pop_size)]
    function2_values2 = [function2(solution2[i])for i in range(0,2*pop_size)]
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
    new_solution= []
    for i in range(0,len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution = [solution2[i] for i in new_solution]
    gen_no = gen_no + 1

 # #Rank based Selection
def selection(Evaluations,SortedEvaluations,pop):
    Evaluations.sort()
    Evaluations
    N=len(Evaluations)
    TotalFitness=(N+1)*N/2

    IndListIndexes,ProbSelection,CumProb,newList,randomy=[],[],[],[],[]
    
    for j in range(N):
        x = random.random()
        randomy.append(x)

        for j in range(N):
      if randomy[j]<CumProb[0]:
          IndListIndexes.append(0)
      else:
        for e in range(N-1):
            if CumProb[e]<randomy[j] and randomy[j]<=CumProb[e+1] :
              #indexes of selected individuals to be accessed in dataframe 
                IndListIndexes.append(e+1)

    newList=SortedEvaluations.iloc[IndListIndexes].index.to_list()
    NewPop=pd.DataFrame(pop.iloc[newList])
    return NewPop
def uniformX(Selected,Pc,pop_size):
    offspring1,offspring2,offsprings,generator=[],[],[],[]

    for g in range(pop_size):
        y = random.random()
        generator.append(y)

    for i in range(0,len(Selected),2):
        offspring1=[]
        offspring2=[]
        # parents
        p1=Selected.iloc[i].to_list()
        p2=Selected.iloc[i+1].to_list()
        p = [p1,p2]    

        crandomy,randomy=[],[]

        if generator[i]<Pc:
          for i in range(len(p[0])):
              offspring1.append(p[randomy[i]][i])
              offspring2.append(p[crandomy[i]][i])
        else:
          for i in range(len(p[0])):
            offspring1.append(p1[i])
            offspring2.append(p2[i])
        offsprings.append(offspring1)
        offsprings.append(offspring2)
    NextGen=pd.DataFrame(offsprings)
    NextGen.index=NextGen.index+1
    NextGen.columns=columns
    NextGen.index.name = 'IndividualID'
    return NextGen
def mutation(NextGeneration, Pm):
    
    a=np.nonzero(mutArr <0.01)

    NG1=NextGeneration.values
    g1=list(NG1[a])

    for x in range(len(g1)):
      b=random.randint(1,5)
      while True:
          if b!=g1[x]:
            break
          else:
            b=random.randint(1,5)
      g1[x]=b

    NG1[a]=g1

    # return population
    NewGeneration=pd.DataFrame(NG1)
    NewGeneration.index=NewGeneration.index+1
    NewGeneration.index.name = 'NewGenID'
    NewGeneration.columns=columns 

    return NewGeneration

def rootedMSE(a,u,new3,NewGeneration,SortedEvaluations):
    x=NewGeneration.iloc[SortedEvaluations.iloc[-1,1]-1][a].values
    y=new3.iloc[u].dropna().values
    return math.sqrt(np.mean((x-y)**2))

def meanAE(a,u,new3,NewGeneration,SortedEvaluations):
    x=NewGeneration.iloc[SortedEvaluations.iloc[-1,1]-1][a].values
    y=new3.iloc[u].dropna().values
    return np.mean(abs(x-y))

usa=[]
for i in range(943):
    u1=new.iloc[i].dropna()
    a=new3.iloc[i].dropna().index
    PearList=correlate(u1,new)
    PEARSON=presentList(PearList,1)
    NeighborInd=topIndex(PEARSON)
    b=new.iloc[NeighborInd][a].any() #.values#.tolist()

    if sum(b.values)>9:
        usa.append(i)
usa[41:44]
u=154       
u1=new.iloc[u]
a=new3.iloc[u].dropna().index
pop_size,Pc,Pm,maxgen=200,0.9,0.01,300
generation,meanGen,meanBest=1,0,0
fit,temp,divide,temp2,RMSE,MAE,tempRMSE,tempMAE=[],[],[],[],[],[],[],[]
times=10
limit=30
 
PearList=correlate(u1,new)
 
PEARSON=presentList(PearList,1)
 
NeighborInd=topIndex(PEARSON)
 
recommendations=findRecommendations2(new,new2,NeighborInd,u)
# recommendations=findRecommendations(u,new,NeighborInd)
movies=recommendations.to_list()
 
 
for i in range(times):
        pop=[np.random.randint(1,6,len(recommendations)) for i in range(pop_size)]
        pop=pd.DataFrame(pop)
        pop.index=pop.index+1
        pop.columns=movies
        pop.index.name = 'IndividualID'
        Neighbors=pd.DataFrame(new.iloc[NeighborInd])
         Evaluations=fitness(pop,Neighbors,recommendations) 
        SortedEvaluations=presentList(Evaluations,0)
        NewPopulation=selection(Evaluations,SortedEvaluations,pop)
        Selected=NewPopulation.loc[:, movies]
        columns=Selected.columns.values
        NextGeneration=uniformX(Selected,Pc,pop_size)
        NewGeneration=mutation(NextGeneration,Pm)
 
        while generation<maxgen :
              generation+=1
 
              Evaluations=fitness(NewGeneration,Neighbors,recommendations) 
              SortedEvaluations=presentList(Evaluations,0)
              NewPopulation=selection(Evaluations,SortedEvaluations,NewGeneration)
 
              Selected=NewPopulation.loc[:, movies]
              columns=Selected.columns.values
                  
              NextGeneration=uniformX(Selected,Pc,pop_size)
              NewGeneration=mutation(NextGeneration,Pm)
              Evaluations=fitness(NewGeneration,Neighbors,recommendations) 
              SortedEvaluations=presentList(Evaluations,0)
              
              temp.append(SortedEvaluations.iloc[-1,0])
              print(generation,SortedEvaluations.iloc[-1,0])
              
              rmse=rootedMSE(a,u,new3,NewGeneration,SortedEvaluations)
              mae=meanAE(a,u,new3,NewGeneration,SortedEvaluations)
 
              print('rmse:',rmse,'mae:',mae)
              tempRMSE.append(rmse)
              tempMAE.append(mae)
 
              if len(temp)>limit :
                if (SortedEvaluations.iloc[-1,0]<temp[-limit]) | ((100*(SortedEvaluations.iloc[-1,0]-temp[-limit])/temp[-limit]) <1e-1):
                    break
        temp2 = np.ones((generation-1), dtype = int)            
        meanGen+=generation
        print('temp,generation,temp2',temp,generation,temp2)
        if i==0:
            fit=temp
            divide=temp2
            RMSE=tempRMSE
            MAE=tempMAE
 
        else:
            fit=[x+y for x,y in zip_longest(fit, temp, fillvalue=0)]
            divide=[x+y for x,y in zip_longest(divide, temp2, fillvalue=0)]
            RMSE=[x+y for x,y in zip_longest(RMSE, tempRMSE, fillvalue=0)]  
            MAE=[x+y for x,y in zip_longest(MAE, tempMAE, fillvalue=0)]
        print('qumulative sum of fitnesses:',fit,'\nqumulative counter of generations:',divide,'\nqumulative RMSE:',RMSE,'\nqumulativeMAE:',MAE)
        temp.sort()
        meanBest+=temp[-1]
        print('sortedtemp,qumulbest,qumulgen:',temp,meanBest,meanGen)
        temp,temp2,tempRMSE,tempMAE=[],[],[],[]
        generation=1
 
meanBest=meanBest/times
meanGen=meanGen/times
fit=[a/b for a,b in zip(fit,divide)]
meanRMSE=[a/b for a,b in zip(RMSE,divide)]
meanMAE=[a/b for a,b in zip(MAE,divide)]
print('fit,meanGen,meanBest',fit,meanGen,meanBest)
plt.plot(fit)
plt.ylabel('Mean Correlation')
plt.xlabel('Generations')
plt.show()
plt.plot(meanRMSE)
plt.ylabel('Mean RMSE')
plt.xlabel('Generations')
plt.show()
plt.plot(meanMAE)
plt.ylabel('Mean MAE')
plt.xlabel('Generations')
plt.show()





