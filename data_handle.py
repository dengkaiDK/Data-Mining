import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

strategy = {1:'delete',2:'mode',3:'relation',4:'similarity'}

class Data:
    def __init__(self,dataset,data_index,strategy=1):
        super().__init__()
        self.dataset = dataset
        self.data_index = data_index
        self.strategy = strategy

    def __len__(self):
        # the number of dataset
        return len(self.dataset)

    def nan_choice(self,choice):
        self.strategy = choice

    def summary(self,item,index):
        dataset = self.dataset[item]
        null_num = dataset[index].isnull().sum()
        print('Nan number of {} column in dataset {}: {}'.format(index,item,null_num))
        slice_index = dataset[index].isnull()
        array = dataset[index].values
        array = array[slice_index == False]
        if index in self.data_index:
            print('five percentile numbers of attribute is ',np.percentile(array,[0,25,50,75,100]))
        
        else:
            a , counts = np.unique(array,return_counts=True)
            print('The number of unique is {}'.format(len(a)))
            diction = {}
            for i in range(len(a)):
                diction[a[i]] = counts[i]
            
            word_freq = sorted(diction.items(),key = lambda d: d[1],reverse = True)
            for j,elem in enumerate(word_freq[:5]):
                word, freq = elem[0],elem[1]
                print('Unique word :{}, frequency :{}, ratio :{:.2f}%'.format(word,freq,freq/len(array)*100))
            
        
        print()
    
    def visualization(self,item,index):
        dataset = self.dataset[item]
        if index not in self.data_index:
            return
        
        array = dataset[index].values
        print()
        print('strategy to handle nan value is: '+strategy[self.strategy])
        if self.strategy == 1:
            array = array[np.isnan(array) == False] # delete nan value            
        
        elif self.strategy == 2:
            array[np.isnan(array) == True] = dataset[index].mode() # use mode to substitute

        elif self.strategy == 3:
            array = self.relation(item,index)   # use normal distribution + curve fit

        elif self.strategy == 4:
            array = self.relation(item,index)   # use mean value

        print('percentile: ',np.percentile(array,[0,25,50,75,100]))
        print('mean value: ',np.mean(array))
        print('std :',np.std(array))
        print()
        plt.figure()
        plt.boxplot(array,labels=[index])
        plt.figure()
        plt.hist(array,bins=20,density=True,color='b',rwidth=0.9)
        plt.title(index)
        plt.show()

    def stat(self,data):
        array = np.array(data)
        mean = np.mean(array)
        std = np.std(array)
        return mean, std

    def polyfit3(self,x,a,b,c,d):
        return a + b * x + c * x**2 + d * x**3

    def logfit(self,x,a,b):
        return a*np.log(x) + b

    def expfit(self,x,a,b):
        return a * np.exp(b/x)

    def relation(self,item,index):
        dataset = self.dataset[item]
        array = dataset[index].values
        row_slice = dataset[index].isnull()
        Y = array[row_slice == False]
        predict = array[row_slice == True]
        # this analysis is specialized for wine dataset , Notice!
        X_index = ['country','province','variety']
        X = []
        pre_X = []

        for ind in X_index:
            temp = dataset[ind].values
            null_index = dataset[ind].isnull()
            a, counts = np.unique(temp[null_index == False],return_counts=True)
            temp[null_index == True] = a[counts.argmax()]  # use high freq word as a substitute
            X.append(temp[row_slice == False])
            pre_X.append(temp[row_slice == True])
        
        X.append(dataset['points'].values[row_slice == False])
        pre_X.append(dataset['points'].values[row_slice == True])

        # basic strategy to model the relationship between columns 'country', 'province', 'variety', 'points' and 'price'
        # primary attributes are 'country', 'province' and 'variety', collect simliar statistic and build a standard distribution
        # acquire a sample from standard distribution, then use points attribute to fit the price.
        # there are some reasons related to the choice of attribute, mainly as a result of unique number and total number, missing value etc.
        # attribute priority: variety > country > province > points
        L = len(predict)
        N = len(Y)
        look_up = {}

        for i in range(N):
            country, province, variety, points = X[0][i], X[1][i], X[2][i], X[3][i]
            price = Y[i]

            if variety not in look_up:
                look_up[variety] = {}
                look_up[variety][country] = {}
                look_up[variety][country][province] = [[price],[points]]
                # mutiple level add element of a dict
            
            elif country not in look_up[variety]:
                look_up[variety][country] = {}
                look_up[variety][country][province] = [[price],[points]]
            
            elif province not in look_up[variety][country]:
                look_up[variety][country][province] = [[price],[points]]
            
            else:
                look_up[variety][country][province][0].append(price)
                look_up[variety][country][province][1].append(points)
            
        #print('build up lookup table done!')
        for i in range(L):
            country, province, variety = pre_X[0][i], pre_X[1][i], pre_X[2][i]
            points = pre_X[3][i]
            mean = 0
            std = 0
            price_list = []
            point_list = []
            if variety in look_up and country in look_up[variety] and province in look_up[variety][country]:
                mean, std = self.stat(look_up[variety][country][province][0])
                price_list.extend(look_up[variety][country][province][0])
                point_list.extend(look_up[variety][country][province][1])

            elif variety in look_up and country in look_up[variety]:
                for elem in look_up[variety][country]:
                    tmp_mean, tmp_std = self.stat(look_up[variety][country][elem][0])
                    mean += tmp_mean
                    std += tmp_std
                    price_list.extend(look_up[variety][country][elem][0])
                    point_list.extend(look_up[variety][country][elem][1])
                mean /= len(look_up[variety][country])
                std /= len(look_up[variety][country])

            elif variety in look_up:
                normalize = 0
                for elem in look_up[variety]:
                    for j in look_up[variety][elem]:
                        tmp_mean, tmp_std = self.stat(look_up[variety][elem][j][0])
                        mean += tmp_mean
                        std += tmp_std
                        price_list.extend(look_up[variety][elem][j][0])
                        point_list.extend(look_up[variety][elem][j][1])
                    normalize += len(look_up[variety][elem])
                
                mean /= normalize
                std /= normalize

            else:
                normalize = 0
                for v in look_up:
                    for elem in look_up[v]:
                        for j in look_up[v][elem]:
                            tmp_mean, tmp_std = self.stat(look_up[v][elem][j][0])
                            mean += tmp_mean
                            std += tmp_std
                            price_list.extend(look_up[v][elem][j][0])
                            point_list.extend(look_up[v][elem][j][1])

                        normalize += len(look_up[v][elem])
                
                mean /= normalize
                std /= normalize
            
            sample = np.random.normal(mean,std)
            price_array = np.array(price_list)
            point_array = np.array(point_list)
            popt,pcov = curve_fit(self.expfit,point_array,price_array - sample)
            predict[i] = self.expfit(points,*popt) + sample
            #predict[i] = mean

        array[row_slice == True] = predict
        return array