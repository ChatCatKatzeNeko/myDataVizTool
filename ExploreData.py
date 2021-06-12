import pandas as pd
import numpy as np


class ExploreData(object):
    '''
    Explore a given pandas dataframe with basic information (shape of the dataset, kinds of variabes and NaN
    percentages).
    
    class attributes are lists of all variables, numerical variables, categorical variables, datetime variables
    and NaN percentage in each variable.
    
    =========
    Parametre
    =========
    data: Python pandas dataframe. 
    
    resetIndex: bool, optional, default = False
                whether to reset data index
                
    autoDetectDatetime: bool, optional, default = False
                        whether to detect possible datetime columns automatically. Warning: if set true, it may
                        take time to complete the detection.
    '''
    
    def __init__(self, data, resetIndex=False, autoDetectDatetime=False):
        self.data = data
        if resetIndex:
            self.data.reset_index(inplace=True, drop=True)
        self.autoDetectDatetime = autoDetectDatetime
        if self.autoDetectDatetime:
            print('autoDetectDatetime set to True.\nWarning: this can be time consuming.\n\n')
        print("The given dataset has %d rows and %d columns." % self.data.shape)
        print("NaNs found in %d columns." % self.data.isna().any(axis=0).sum())
        
        self.vars = list(self.data.columns)
        self.numerical_vars = []
        self.categorical_vars = []
        self.datetime_vars = []
        self.nan_perc = []
        
    def __call__(self):
        if self.autoDetectDatetime:
            print('Detecting possible datetime columns...\nWarning: this can be time consuming.')
            self.detectDatetimeVariables()
        self.getNumericalVariables()
        self.getDatetimeVariables()
        self.getCategoricalVariables()
        self.summary = []
        if len(self.numerical_vars) > 0:
            self.summary.append(self.data.loc[:,self.numerical_vars].describe())
        if len(self.categorical_vars) > 0:
            self.summary.append(self.data.loc[:,self.categorical_vars].describe())
        self.getNaNInfo()
        print("\n------------------------------\nNaN percentage of each column:\n------------------------------\n")
        for (col, perc) in self.nan_perc:
            print("\t%s: %.2f%%" % (col, perc*100))
     
     
    # tested
    def getNumericalVariables(self):
        '''
        fill the list of numerical variables
        '''
        self.numerical_vars = list(self.data.select_dtypes(include=[np.number]).columns)
    
    # tested
    def getCategoricalVariables(self):
        '''
        fill the list of categorical variables
        '''
        self.categorical_vars = list(set(self.vars) - set(self.numerical_vars) - set(self.datetime_vars))
    
    # tested
    def detectDatetimeVariables(self):
        mask = self.data.astype(str).apply(lambda x : x.str.match(r'(\d{2,4}-\d{2}-\d{2,4})+').all())
        self.data.loc[:,mask] = self.data.loc[:,mask].apply(pd.to_datetime)
    
    # tested
    def getDatetimeVariables(self):
        '''
        fill the list of datetime variables
        '''
        self.datetime_vars = list(self.data.select_dtypes(include=[np.datetime64, np.timedelta64, 'datetime', 'datetime64', 'timedelta', 'datetimetz','timedelta64']).columns)
     
    # tested
    def getNaNInfo(self):
        '''
        fill the list of NaN percentage of each column
        '''
        nbNaN = [self.data.loc[:,col].isna().sum() for col in self.data.columns]
        self.nan_perc = [(self.data.columns[i], nb/self.data.shape[0]) for i,nb in enumerate(nbNaN) if nb > 0]
        
        
    # tested
    def createGroupedDf(self, catCols, agg, data=None):
        '''
        Return grouped dataframe with given categorical variable and aggregation method
        
        ======
        INPUTS
        ======
        catCols: list of strings
                 names of the column by which data will be grouped. Must be in the list of categorical_vars.
                     
        agg: string or function or dictionary
             aggregation method. 
             If a string is given, it should be in {"sum", "mean", "max", "min", "count"}.
             If a function is given, it's the user who should take care of the NaN values.
             If a dictionary is given, it must be the form of {column name(string): aggregation method(string/function)}
             
        data: pandas dataframe, optional, default = None
              data to be plot. If None, then the class atribute data will be used.
        
        ======
        OUTPUT
        ======
        resampled dataframe
        
        '''
        if data is None:
            data = self.data
         
        data.reset_index(inplace=True, drop=True)
        
        for catCol in catCols:
            if catCol not in self.categorical_vars:
                raise AssertionError("catCol must be in the list of categorical_vars.")
            
        if type(agg) is str:
            if agg == 'sum':
                return data.groupby(by=catCols).sum()
            elif agg == 'mean':
                return data.groupby(by=catCols).mean()
            elif agg == 'max':
                return data.groupby(by=catCols).max()
            elif agg == 'min':
                return data.groupby(by=catCols).min()
            elif agg == 'count':
                return data.groupby(by=catCols).count()
            else:
                raise ValueError('agg can either be a string in {"sum", "mean", "max", "min", "count"} or be a function.')
        
        elif type(agg) is type(lambda x:x): # if agg is a function
            return data.groupby(by=catCols).apply(agg)
        
        elif type(agg) is dict:
            return data.groupby(by=catCols).agg(agg)
        else:
            raise ValueError('agg can be a string in {"sum", "mean", "max", "min", "count"}, or a function, or a dictionary.')
        
    
    # tested
    def createResampledDf(self, freq, datetimeCol, agg, data=None):
        '''
        Return resampled dataframe with given frequency and aggregation method
        
        ======
        INPUTS
        ======
        freq: string, 
              frequency value. "S" for second, "T" for minute, "H" for hour, "D" for day, "W" for week, 
              "M" for month, "Y" for year etc..
              
        datetimeCol: string
                     name of the column on which data will be resampled. Must be in the list of datetime_vars.
                     
        agg: string or function or dictionary
             aggregation method. 
             If a dictionary is given, it must be the form of {column name(string): aggregation method(string/function)}
             If a function is given, it's the user who should take care of the NaN values.
        
        data: pandas dataframe, optional, default = None
              data to be plot. If None, then the class atribute data will be used.
        ======
        OUTPUT
        ======
        resampled dataframe
        
        '''
        if data is None:
            data = self.data
         
        data.reset_index(inplace=True, drop=True)
        
        if datetimeCol not in self.datetime_vars:
            raise AssertionError("datetimeCol must be of type datetime.")
            
        if type(agg) is str:
            if agg == 'sum':
                return data.resample(freq, on=datetimeCol).sum()
            elif agg == 'mean':
                return data.resample(freq, on=datetimeCol).mean()
            elif agg == 'max':
                return data.resample(freq, on=datetimeCol).max()
            elif agg == 'min':
                return data.resample(freq, on=datetimeCol).min()
            else:
                raise ValueError('agg can either be a string in {"sum", "mean", "max", "min"} or be a function.')
        
        elif type(agg) is type(lambda x:x): # if agg is a function
            return data.resample(freq, on=datetimeCol).asfreq().apply(agg)
       
        elif type(agg) is dict:
            return data.resample(freq, on=datetimeCol).agg(agg)
        
        else:
            raise ValueError('agg can either be a string in {"sum", "mean", "max", "min"} or be a function or an aggregation dictionary.')
      
    
    # tested
    def detectAbnormalities(self, rules):
        '''
        detect simple abnormalities.
        
        =====
        INPUT
        =====
        rules: dictionary
               Keys are column names, values are rules, i.e., the ways that the values should be. 
               Rules can be a tuple of (string, value) or (string, lower, upper) or functions that take the rows of
               the dataframe as input. 
               For e.g., rules = {'var1': ('>',0), 
                                  'var3': lambda x: x.var2 * x.var5, 
                                  'var7': ('=',1),
                                  'var0': ('in', 0, 1)} 
               defines 4 rules:
                   1. var1 ought to be strictly greater than 0
                   2. var3 is the product of var2 and var5   
                   3. var7 ought to be 1
                   4. var0 is in [0,1]
                   
        ======
        OUTPUT
        ======
        list of tuples where rules are not statisfied
        '''
        
        abnormalities = []
        for i,var in enumerate(rules):
            if type(rules[var]) is tuple and len(rules[var]) == 2:
                if rules[var][0] == '>':
                    abnormalities.append(('rule_%d'%(i+1), self.data.loc[self.data.loc[:, var] <= rules[var][1]]))
                elif rules[var][0] == '>=':
                    abnormalities.append(('rule_%d'%(i+1), self.data.loc[self.data.loc[:, var] < rules[var][1]]))
                    
                elif rules[var][0] == '<':
                    abnormalities.append(('rule_%d'%(i+1), self.data.loc[self.data.loc[:, var] >= rules[var][1]]))
                elif rules[var][0] == '<=':
                    abnormalities.append(('rule_%d'%(i+1), self.data.loc[self.data.loc[:, var] > rules[var][1]]))
                    
                elif rules[var][0] == '=':
                    abnormalities.append(('rule_%d'%(i+1), self.data.loc[self.data.loc[:, var] != rules[var][1]]))
                
                else:
                    raise ValueError("Comparaison signs can only be in {'>', '>=', '=', '<=', '<'}.")
            
            elif type(rules[var]) is tuple and len(rules[var]) == 3:  
                if rules[var][0] == 'in':
                    abnormalities.append(('rule_%d'%(i+1), self.data.loc[(self.data.loc[:, var] < rules[var][1]) | (self.data.loc[:, var] > rules[var][2])]))
                
                elif rules[var][0] == 'out':
                    abnormalities.append(('rule_%d'%(i+1), self.data.loc[(self.data.loc[:, var] > rules[var][1]) & (self.data.loc[:, var] < rules[var][2])]))
                
                else:
                    raise ValueError("Valid choices are {'in', 'out'}.")
                    
            elif type(rules[var]) is type(lambda x: x): # if the rule is a function
                abnormalities.append(('rule_%d'%(i+1), self.data.loc[self.data.loc[:,var] != self.data.apply(rules[var], axis=1)]))
            
            else:
                raise ValueError("Invalid rules.")
            
        return abnormalities

    
    def detectProblematicColumns(self, threshold=0.75):
        '''
        create class attributes: 
        sparseCols: columns where NaNs > threshold
        constantCols: columns where values are constant
        
        =====
        INPUT
        =====
        threshold: float in [0,1], default = 0.75
                   if a column's NaN rate surpass threshold, it will be considered as sparse.
        '''
        self.sparseCols = [col for (col, perc) in self.nan_perc if perc > threshold]
        self.constantCols = [col for col in self.numerical_vars if self.data.loc[:,col].std() == 0]
        self.constantCols += [col for col in self.categorical_vars if self.data.loc[:,col].nunique() == 1]
        
    def dropColumns(self, cols, inplace=True):
        '''
        drop given columns
        
        ======
        INPUTS
        ======
        cols: list of strings
              the list of columns to be dropped
              
        inplace: bool, default = True
                 whether to modify the dataframe in place
        '''
        self.data.drop(cols, axis=1, inplace=inplace)
        
        if inplace:
            self.numerical_vars = list(set(self.numerical_vars) - set(cols))
            self.categorical_vars = list(set(self.categorical_vars) - set(cols))
            self.datetime_vars = list(set(self.datetime_vars) - set(cols))
