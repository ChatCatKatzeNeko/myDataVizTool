'''
Author: Siyun WANG
'''
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

from ExploreData import ExploreData

class BasicStatisticPlots(object):
    '''
    Make basic statistic plots for data visualisation
    
    ==========
    Parametres
    ==========
    expData: ExploreData object
    '''
    def __init__(self, expData):
        self.explorer = expData
        self.explorer()
        self.data = expData.data
        self.numerical_vars = expData.numerical_vars
        self.categorical_vars = expData.categorical_vars
        self.datetime_vars = expData.datetime_vars
        self.nb_rows = self.data.shape[0]
        self.nb_cols = self.data.shape[1]
        
    # tested    
    def corrMatPlot(self, data=None, annot=True, threshold=None):
        '''
        plot correlation matrix
        
        =====
        INPUT
        =====
        data: pandas dataframe, optional, default = None
              data to be plot. If None, then the class attribute data will be used.
        
        annot: boolean, optional, default = True
               whether to print the exact value of each element in the correlation matrix
               
        threshold: float between 0 and 1, default = None
                   if given, all cells having absolut correlation below the value will be masked
        '''
        if data is None:
            data = self.data
            
        corr = data.loc[:, self.numerical_vars].corr()

        mask = np.triu(np.ones_like(corr, dtype=np.bool))
	
        if threshold is not None:
            mask[np.where(np.abs(corr) < threshold)] = True
        
        plt.figure(figsize=(16,12))
        sns.heatmap(data=corr, vmin=-1, vmax=1, cmap='RdBu_r', 
                    annot=annot, cbar=True, square=True, mask=mask)
        plt.title("correlation matrix")
        plt.show()    
     
    # tested
    def distPlot(self, col, drop_outliers=True, bins=None, data=None, lab=None):
        '''
        plot histogram of given variable
        
        ======
        INPUTS
        ======
        col: string
             variable's column name.
    
        drop_outliers: bool, default = True
                       whether to drop datapoints who fall 3 standard deviations away from the average.
        
        bins: int or list, default = None
              seaborn distplot's bin parametre.
              
        data: pandas dataframe, optional, default = None
              data to be plot. If None, then the class attribute data will be used.
              
        lab: string, optional, default = None
             axis label. if None, column name will be used
        '''
        if data is None:
            data = self.data
        
        if lab is None:
            lab = col
        plt.figure(figsize=(16,8))
        if drop_outliers:
            sns.distplot(a=data.loc[(abs((data.loc[:,col]-data.loc[:,col].mean())/data.loc[:,col].std())<3), col], kde=False, norm_hist=True)

        else:
            sns.distplot(a=data.loc[:, col].dropna(), bins=bins, kde=False, norm_hist=True)

        plt.grid()
        plt.title('distribution of %s' % lab)
        plt.xlabel(lab)
        plt.ylabel('frequency')
        plt.xticks(rotation=-60)

        plt.show()
     

    def checkCorrelation(self, threshold, drop_outliers=True, data=None):
        '''
        plot scatter plots of highly correlated features
        
        =====
        INPUT
        =====
        drop_outliers: bool, default = True
              whether to drop datapoints who fall 3 standard deviations away from the average.
        
        threshold: float between 0 and 1, default = None
                   if given, all cells having absolut correlation below the value will be masked
        
        data: pandas dataframe, optional, default = None
              data to be plot. If None, then the class attribute data will be used.
        
        '''
        if data is None:
            data = self.data
        corr = data.loc[:, self.numerical_vars].corr().values
        corr[np.triu_indices_from(corr)] = 0 # mask upper triangle
        mask = np.where((np.abs(corr) >= threshold) & (np.abs(corr) < 1))

        for c1, c2 in zip(mask[0], mask[1]):
            col1 = self.numerical_vars[c1]
            col2 = self.numerical_vars[c2]
            print("==================")
            print("correlation between %s and %s: %.4f" % (col1, col2, corr[c1,c2]))
            self.scatterPlot(col1, col2,
                             drop_outliers=drop_outliers, data=data)
            print("\n\n")
    
    # tested
    def scatterPlot(self, col1, col2, col3=None, drop_outliers=True, data=None, lab1=None, lab2=None):
        '''
        plot scatter plot for given variables
        
        ======
        INPUTS
        ======
        col1: string
              x variable's column name.
              
        col2: string
              y variable's co lumn name.
              
        col3: string, optional, default = None
              hue variable's column name. If a third variable is provided, the points will be distinguished by this variable, otherwise scatter plot with histograms of each x,y variable is plotted. Note that the hue variable should be categorical.
              
        drop_outliers: bool, default = True
                       whether to drop datapoints who fall 3 standard deviations away from the average.
        
        data: pandas dataframe, optional, default = None
              data to be plot. If None, then the class atribute data will be used.
              
        lab1, lab2: strings, optional, default = None
                    axis labels. if None, column names will be used
        '''
        if data is None:
            data = self.data
        if lab1 is None:
            lab1 = col1
        if lab2 is None:
            lab2 = col2
        
        if col3 is not None:
            if data.loc[:, col3].nunique() > 10:
                raise ValueError("Too many labels in %s, please flag or re-group them." % col3)

            plt.figure(figsize=(16,8))
            if drop_outliers:
                sns.scatterplot(x=col1, y=col2, data=data.loc[(abs((data.loc[:,col1]-data.loc[:,col1].mean())/data.loc[:,col1].std())<3)], 
                                hue=col3, 
                                #style=col3
                               )
            else:
                sns.scatterplot(x=col1, y=col2, data=data, 
                                hue=col3, 
                                #style=col3
                               )
            plt.xlabel(lab1)
            plt.ylabel(lab2)
            plt.xticks(rotation=-60)
            plt.title('scatter plot of %s vs %s' % (lab1, lab2))
            plt.grid()
            plt.show()
        
        else:
            if drop_outliers:
                sns.jointplot(x=col1, y=col2, 
                              data=data.loc[(abs((data.loc[:,col1]-data.loc[:,col1].mean())/data.loc[:,col1].std())<3)],
                              height=10)
            else:
                sns.jointplot(x=col1, y=col2, data=data,
                              height=10)
            plt.show()
     
    # tested
    def scatterPlot_1vsRest(self, col, variables, hue=None, drop_outliers=False, asX=True, data=None):
        '''
        plot scatter plots for given variables
        
        ======
        INPUTS
        ======
        col: string
             variable's column name.
             
        variables: array-like object 
                   contains the variables to be plotted as an other axis 
                   
        hue: string, optional, default = None
             hue variable's column name. If provided, the points will be distinguished by this variable, otherwise scatter plot with histograms of each x,y variable is plotted. Note that the hue variable should be categorical.
             
        drop_outliers: bool, default = True
                       whether to drop datapoints who fall 3 standard deviations away from the average.
                       
        asX: bool, default = True
             True if the "col" should be the x variable and the other variables in "variables" are the y variable, False vice-versa
             
        data: pandas dataframe, optional, default = None
              data to be plot. If None, then the class atribute data will be used.
        '''
            
        variables = list(variables)
        if col in variables:
            variables.remove(col)
        if asX:
            for var in variables:
                self.scatterPlot(col, var, hue, drop_outliers=drop_outliers, data=data)
        else:
            for var in variables:
                self.scatterPlot(var, col, hue, drop_outliers=drop_outliers, data=data)

    
    # tested
    def piePlot(self, cols, agg, col_y=None, data=None):
        '''
        create a grouped dataframe by the given categorical variable and plot a pie
        
        ======
        INPUTS
        ======
        cols: list of strings
              variable names by which the dataframe is to be grouped.
    
        agg: ExploreData.createGroupedDf's agg parametre
        
        col_y: string, optional, default = None
               the target column name to be plotted. If not given, the first one in cols is taken.
               
        data: pandas dataframe, optional, default = None
              data to be plot. If None, then the class attribute data will be used.
        '''
            
        grouped = self.explorer.createGroupedDf(cols, agg, data=data)
        if grouped.index.nlevels > 2:
            raise ValueError("Too many levels of index. Allowed: 2; Recieved: %d" % grouped.index.nlevels)
        
        # if the grouped dataframe has 2 levels of index
        elif grouped.index.nlevels == 2:
            
            # for e.g., a grouped dataframe obtained by grouping variables [v1, v2] and aggregated by summation 
            # over the variable v3
            # the grouped dataframe may look like this:
            # v1   v2   v3_agg
            # ------------------
            # A    a    10
            #      -------------
            #      b    5
            #      -------------
            #      d    5
            # ------------------
            # B    b    10
            #      -------------
            #      c    15
            # we want to plot 2 plots for A and B, a pie in such a plot is anything in {a, b, c, d} (values of v2),
            # the size of a pie is defined by the corresponding value. 
            # Precisely, for the pie plot A, the pie a occupies 50% of the chart, the pie b and the pie d take each
            # one of both 25% of the chart
            
            for ind in grouped.index.get_level_values(cols[0]).unique(): 
                print(cols[0] + ': ' + str(ind))
                plt.figure()
                tmp = grouped.loc[ind]
                tmp.plot(y=col_y, subplots=True, kind='pie', figsize=(10,10), legend=False)
                plt.show()
                
        # if the grouped dataframe has single level index, plot simple pie plot by index
        elif grouped.index.nlevels == 1:
            plt.figure()
            grouped.plot(y=col_y, subplots=True, kind='pie', figsize=(10,10), legend=False)
            plt.show()
        
        else:
            raise ValueError("Invalid indexing")
     
    # 
    def boxPlot(self, col1, col2, col3=None, drop_outliers=True, plotEach=False, data=None):
        '''
        plot scatter plot for given variables
        
        ======
        INPUTS
        ======
        col1: string
              x variable's column name. Should be categorical.
              
        col2: string
              y variable's column name.
        
        col3: string, optional, default = None
              hue variable's column name. If a third variable is provided, the points will be distinguished by this variable.
                     
        drop_outliers: bool, default = True
                       whether to drop datapoints who fall 3 standard deviations away from the average.
                       
        plotEach: bool, default = False
                  whether to plot each point (if set to be True, it can be slow if the amount of data is huge)
        
        data: pandas dataframe, optional, default = None
              data to be plot. If None, then the class atribute data will be used.
        '''
        if data is None:
            data = self.data
        
        data.reset_index(inplace=True, drop=True)
        
        if col1 not in self.categorical_vars:
            raise ValueError("col1 should be a categorical variable.")
            
        plt.figure(figsize=(16,8))
        if drop_outliers:
            sns.boxplot(x=col1, y=col2, hue=col3, data=data.loc[(abs((data.loc[:,col2]-data.loc[:,col2].mean())/data.loc[:,col2].std())<3)])
            if plotEach:
                sns.stripplot(x=col1, y=col2, hue=col3, data=data.loc[(abs((data.loc[:,col2]-data.loc[:,col2].mean())/data.loc[:,col2].std())<3)], 
                              dodge=True, alpha=0.5)
        else:
            sns.boxplot(x=col1, y=col2, hue=col3, data=data)
            if plotEach:
                sns.stripplot(x=col1, y=col2, hue=col3, data=data, 
                              dodge=True, alpha=0.5)
        plt.grid()
        plt.title('box plot of %s with respect to %s' % (col2, col1))
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.xticks(rotation=-60)

        plt.show()

    # 
    def boxPlot_1vsRest(self, col, variables, hue=None, drop_outliers=True, plotEach=False, data=None):
        '''
        plot box plots for given variables
        
        ======
        INPUTS
        ======
        col: string
             y variable's column name.
             
        variables: array-like object 
                   contains the variables to be plotted as x. Variables should be in the categorical variables. 
                   
        hue: string, defautl = None
             hue variable's column name. If provided, the points will be distinguished by this variable, otherwise box plots with histograms of each x,y variable are plotted.
             
        drop_outliers: bool, default = True
                       whether to drop datapoints who fall 3 standard deviations away from the average.
                       
        plotEach: bool, default = False
                  whether to plot each point (if set to be True, it can be slow if the amount of data is huge)
                  
        data: pandas dataframe, optional, default = None
              data to be plot. If None, then the class atribute data will be used.
        '''
        if data is None:
            data = self.data
            
        for var in variables:
            if data.loc[:,var].nunique() > 10:
                print("Number of unique values of %s is greater than 10, pleas flag or regroup them for better visualisation." % var)
            else:    
                self.boxPlot(var, col, drop_outliers=drop_outliers, plotEach=plotEach, data=data)
            

    # tested
    def timeSeriesPlot(self, datetimeCol, cols, freq=None, agg=None, data=None):
        '''
        plot time series curves
        
        ======
        INPUTS
        ======
        datetimeCol: string
                     datetime variable's name
                 
        cols: list of strings
              variable to be plotted over datetimeCol.
              
        freq: string, optional, default = None
              frequency value for resampling data. "S" for second, "T" for minute, "H" for hour, "D" for day, "W" for week, "M" for month, "Y" for year etc..
              
        agg: string or function, optional, default = None
             aggregation method for resampling data. If a function is given, it's the user who should take care of the NaN values. If None, no resampling will be peformed.
             
        data: pandas dataframe, optional, default = None
              data to be plot. If None, then the class atribute data will be used.
        '''
        if data is None:
            data = self.data
            
        if datetimeCol not in self.datetime_vars:
            raise ValueError("datetimeCol should be a datetime variable.")

        plt.figure(figsize=(16,8))
        if agg is None:
            for col in cols:
                plt.plot(data.loc[:, datetimeCol], data.loc[:,col], alpha=0.5, label=col)

        else:
            data.reset_index(inplace=True, drop=True)
            df_plot = self.explorer.createResampledDf(freq, datetimeCol, agg, data=data)
            for col in cols:
                plt.plot(df_plot.index, df_plot.loc[:,col], alpha=0.5, label=col)
        
        plt.grid()
        plt.title('evolution of variable(s) over time, frequency %s' % freq)
        plt.xlabel('time')
        plt.ylabel('quantity')
        plt.legend(loc=0)
        plt.xticks(rotation=-60)

        plt.show()
    
    # tested
    def timeSeriesPlot_twinX(self, datetimeCol, cols1, cols2, freq=None, agg=None, data=None):
        '''
        plot 2 time series curves sharing x axis and having seperated y axes for each curve
        
        ======
        INPUTS
        ======
        datetimeCol: string
                     datetime variable's name
                 
        cols1,2: lists
                variables to be plotted over datetimeCol.
              
        freq: string, optional, default = None
              frequency value for resampling data. "S" for second, "T" for minute, "H" for hour, "D" for day, "W" for week, "M" for month, "Y" for year etc..
              
        agg: string or function, optional, default = None
             aggregation method for resampling data. If a function is given, it's the user who should take care of the NaN values. If None, no resampling will be peformed.
             
        data: pandas dataframe, optional, default = None
              data to be plot. If None, then the class atribute data will be used.
        '''
        if data is None:
            data = self.data
            
        if datetimeCol not in self.datetime_vars:
            raise ValueError("datetimeCol should be a datetime variable.")
            
        if agg is None:
            df_plot = data
            t = data.loc[:,datetimeCol]
        else:
            data.reset_index(inplace=True, drop=True)
            df_plot = self.explorer.createResampledDf(freq, datetimeCol, agg, data=data)
            t = df_plot.index

        colours1 = sns.color_palette("PuBu_r", n_colors=len(cols1))
        colours2 = sns.color_palette("YlOrRd_r", n_colors=len(cols2))
        
        fig, ax1 = plt.subplots(figsize=(16,8))
        
        for i,col1 in enumerate(cols1):
            s1 = df_plot.loc[:, col1]
            ax1.plot(t, s1, ':', color=colours1[i], alpha=0.8, linewidth=3, label=col1)
        
        ax1.set_xlabel('time_axis')
        ax1.legend(loc=2)

        # Make the y-axis label, ticks and tick labels match the line colour.
        ax1.set_ylabel(col1, color='steelblue')
        ax1.tick_params('y', colors='steelblue')
        ax1.grid(color='steelblue', alpha=0.4, axis='y', linestyle='--')

        ax2 = ax1.twinx()
        for i,col2 in enumerate(cols2):
            s2 = df_plot.loc[:,col2]
            ax2.plot(t, s2, color=colours2[i], alpha=0.7, label=col2)
        ax2.set_ylabel(col2, color='orange')
        ax2.tick_params('y', colors='orange')
        ax2.grid(color='orange', alpha=0.4, axis='y', linestyle='-.')
        ax2.legend(loc=1)

        fig.tight_layout()

        plt.title('Evolution of variables by time')
        plt.show()
        
    
    # tested
    def timeSeriesDecomposition(self, datetimeCol, col, freq=None, agg=None, data=None):
        '''
        decompose a time series in to y(x) = trend + seasonality + noise and plot each component
        ======
        INPUTS
        ======
        datetimeCol: string
                     datetime variable's name
                 
        col: string
             variable to be plotted over datetimeCol.
              
        freq: string, optional, default = None
              frequency value for resampling data. "S" for second, "T" for minute, "H" for hour, "D" for day, "W" for week, "M" for month, "Y" for year etc..
              
        agg: string or function, optional, default = None
             aggregation method for resampling data. If a function is given, it's the user who should take care of the NaN values. If None, no resampling will be peformed.
             
        data: pandas dataframe, optional, default = None
              data to be plot. If None, then the class atribute data will be used.
        ======
        OUTPUT
        ======
        the result of the decomposition
        '''
        if data is None:
            data = self.data
            
        if datetimeCol not in self.datetime_vars:
            raise ValueError("datetimeCol should be a datetime variable.")
            
        if agg is None:
            df = data
        else:
            data.reset_index(inplace=True, drop=True)
            df = self.explorer.createResampledDf(freq, datetimeCol, agg, data=data)
            
        series = df.loc[:,col]
        result = seasonal_decompose(series, model='additive')
        fig, (ax0,ax1,ax2,ax3) = plt.subplots(4,1, figsize=(35,20))
        result.observed.plot(ax=ax0)
        result.trend.plot(ax=ax1)
        result.seasonal.plot(ax=ax2)
        result.resid.plot(ax=ax3)
        
        plt.show()
        
        return result

    
    # tested
    # Quite special a function, I can't see how it can be generalised to other projects of different kinds...
    def timeSeriesPlot_folded(self, datetimeCol, groupbyCols, plotCol, foldFreq, 
                              fixYLim=False, inPercentage=False, percentageOn=None, cumulateSum=False, 
                              freq=None, agg=None, data=None):
        '''
        plot time series curves of one variable over a same period
        
        ======
        INPUTS
        ======
        datetimeCol: string
                     datetime variable's name
                 
        groupbyCols: list of strings
                     variables to be grouped.
                    
        plotCol: stirng
                 variable to be plotted.
        
        foldFreq: string
                  the frequency that distinguishes the curves, must be longer than the frequency for resampling data. Availble frequencies are {'W', 'M', 'Y'}. For e.g., if one wants to study the average temperature of each week over years, then the foldFreq will be "Y" for year while the freq for resampling data will be "W" for week.
        
        fixYLim: bool, default = False
                 whether to fix y limits as the same for all figures.
                 
        inPercentage: bool, default = False
                      whether to convert the variable to be plotted into percentages.
        
        percentageOn: string, default = None
                      Column name, only applied when inPercentage is set to True. If given, a sum of the plotCol will be calculated stratified by the given column and the resampled datetime column, otherwise the sum is calculated only on the stratified datetime.
                       
        cumulateSum: bool, default = False
                     whether to plot the variable in its cumulated sum. Note that if set True, fixYLim is automatically set to False.
                     
        freq: string, optional, default = None
              frequency value for resampling data. Available frequencies here are {'D', 'W', 'M'} for day, week and month respectively.
              
        agg: dictionary or function, optional, default = None
             aggregation method for resampling data. If a function is given, it's the user who should take care of the NaN values. If None, no resampling will be peformed.
             
        data: pandas dataframe, optional, default = None
              data to manipulate with. If None, then the class atribute data will be used.
        '''
        
        if data is None:
            data = self.data
        data.reset_index(inplace=True, drop=True)
        
        if datetimeCol not in self.datetime_vars:
            raise ValueError("datetimeCol should be a datetime variable.")

        # group dataframe
        df_plot = data.groupby(by=groupbyCols).resample(freq, on=datetimeCol)

        # aggregate dataframe by user-defined method
        if type(agg) is type(lambda x:x): # if agg is a function
            df_plot = df_plot.apply(agg)
        elif type(agg) is dict:
            df_plot = df_plot.agg(agg)
        else:
            raise ValueError('agg can either be a function or an aggregation dictionary.')
      
        if type(df_plot) is pd.Series:
            df_plot = pd.DataFrame(df_plot)
            df_plot.columns = [plotCol]
        
        df_plot.reset_index(level=datetimeCol, inplace=True)
        
        if inPercentage: 
            if percentageOn is None:
                total = data.resample(freq, on=datetimeCol).agg({plotCol:'sum'})
            else:
                total = data.groupby(by=percentageOn).resample(freq, on=datetimeCol).agg({plotCol:'sum'})
                
            total.columns = ['SumOfPlotCol']
            df_plot = df_plot.join(total, on=datetimeCol)
            df_plot.loc[:, plotCol] = df_plot.loc[:, plotCol].div(df_plot.SumOfPlotCol)
         
        # define plt.ylim
        bottom, top = df_plot.loc[:, plotCol].min()*0.95, df_plot.loc[:, plotCol].max()*1.05
        
        # define x-axis' time unity
        if freq == 'W':
            df_plot['unity'] = df_plot.loc[:,datetimeCol].dt.week
        elif freq == 'M':
            df_plot['unity'] = df_plot.loc[:,datetimeCol].dt.month
        elif freq == 'D':
            df_plot['unity'] = df_plot.loc[:,datetimeCol].dt.day
        else:
            raise ValueError("Available 'freq' frequencies are {'D','W','M'}")

        # define period of the fold
        if foldFreq == 'W':
            df_plot['foldFreq'] = df_plot.loc[:,datetimeCol].dt.week
        elif foldFreq == 'M':
            df_plot['foldFreq'] = df_plot.loc[:,datetimeCol].dt.month
        elif foldFreq == 'Y':
            df_plot['foldFreq'] = df_plot.loc[:,datetimeCol].dt.year
        else:
            raise ValueError("Available 'foldFreq' frequencies are {'W','M','Y'}")

        # if the user wants the curve to be in cumulated sum (special case, only make sense when aggregation is a sum)
        if cumulateSum:
            fixYLim = False
            # if the filter is of ordre 1
            if len(groupbyCols) == 1:
                for ind in df_plot.index.unique():
                    plt.figure(figsize=(18,6))
                    x_bottom, x_top = df_plot.unity.min(), df_plot.unity.max()
                    for ff in df_plot.foldFreq.unique():
                        tmp = df_plot.loc[ind,:]
                        plt.plot(tmp.loc[tmp.foldFreq == ff].unity.values, tmp.loc[tmp.foldFreq == ff, plotCol].cumsum(), '-*',
                                 alpha=0.5, label=ff)

                    if fixYLim:
                        plt.ylim(bottom, top)

                    plt.xlim(x_bottom, x_top)
                    plt.legend(loc=0)
                    plt.grid()
                    plt.title('Evolution of %s resampled by %s [%s: %s]' % (plotCol, freq, groupbyCols[0], ind))
                    plt.show()
            
            # if a second ordre filter is applied
            elif len(groupbyCols) == 2:
                for ind0 in df_plot.index.get_level_values(groupbyCols[0]).unique():
                    TMP = df_plot.loc[ind0]
                    x_bottom, x_top = TMP.unity.min(), TMP.unity.max()
                    print('\==========================================')
                    print(groupbyCols[0] + ": " + ind0)
                    for ind in TMP.index.unique():
                        plt.figure(figsize=(18,6))
                        for ff in TMP.foldFreq.unique():
                            tmp = TMP.loc[ind,:]
                            plt.plot(tmp.loc[tmp.foldFreq == ff].unity.values, tmp.loc[tmp.foldFreq == ff, plotCol].cumsum(), 
                                     '-*', alpha=0.5, label=ff)

                        if fixYLim:
                            plt.ylim(bottom, top)

                        plt.xlim(x_bottom, x_top)
                        plt.legend(loc=0)
                        plt.grid()
                        plt.title('Evolution of %s resampled by %s [%s: %s]' % (plotCol, freq, groupbyCols[1], ind))
                        plt.show()
            
            # currently does not support higher ordre filter, raise error message 
            else:
                raise ValueError("Too many levels of index. Allowed: 2; Recieved: %d" % len(groupbyCols))
        
        # if curves are not in cumulated sum
        else:
            # if the filter is of ordre 1
            if len(groupbyCols) == 1:
                for ind in df_plot.index.unique():
                    plt.figure(figsize=(18,6))
                    x_bottom, x_top = df_plot.unity.min(), df_plot.unity.max()
                    for ff in df_plot.foldFreq.unique():
                        tmp = df_plot.loc[ind,:]
                        plt.plot(tmp.loc[tmp.foldFreq == ff].unity.values, tmp.loc[tmp.foldFreq == ff, plotCol], '-*',
                                 alpha=0.5, label=ff)

                    if fixYLim:
                        plt.ylim(bottom, top)

                    plt.xlim(x_bottom, x_top)
                    plt.legend(loc=0)
                    plt.grid()
                    plt.title('Evolution of %s resampled by %s [%s: %s]' % (plotCol, freq, groupbyCols[0], ind))
                    plt.show()
            
            # if a second ordre filter is applied
            elif len(groupbyCols) == 2:
                for ind0 in df_plot.index.get_level_values(groupbyCols[0]).unique():
                    TMP = df_plot.loc[ind0]
                    x_bottom, x_top = TMP.unity.min(), TMP.unity.max()
                    print('==========================================')
                    print(groupbyCols[0] + ": " + ind0)
                    for ind in TMP.index.unique():
                        plt.figure(figsize=(18,6))
                        for ff in TMP.foldFreq.unique():
                            tmp = TMP.loc[ind,:]
                            plt.plot(tmp.loc[tmp.foldFreq == ff].unity.values, tmp.loc[tmp.foldFreq == ff, plotCol], '-*',
                                     alpha=0.5, label=ff)

                        if fixYLim:
                            plt.ylim(bottom, top)
                            
                        plt.xlim(x_bottom, x_top)
                        plt.legend(loc=0)
                        plt.grid()
                        plt.title('Evolution of %s resampled by %s [%s: %s]' % (plotCol, freq, groupbyCols[1], ind))
                        plt.show()
            
            # currently does not support higher ordre filter, raise error message 
            else:
                raise ValueError("Too many levels of index. Allowed: 2; Recieved: %d" % len(groupbyCols))
