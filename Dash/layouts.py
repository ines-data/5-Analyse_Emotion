import dash_core_components as dcc
import dash_html_components as html
import dash_core_components as dcc
import dash_html_components as html
from app import app
from urllib.parse import quote
import pandas as pd 
import numpy as np
import plotly.graph_objs as go
import dash_table
from dash.dependencies import Output, Input
import plotly.graph_objects as go  
# import figure factory
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, ComplementNB, BernoulliNB
from sklearn.decomposition import FastICA, KernelPCA, TruncatedSVD, SparsePCA, NMF, FactorAnalysis, LatentDirichletAllocation
#from nltk.corpus import stopwords

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#################################################### Analyse fichier 1 ##############################################
df1_n = pd.read_csv('Emotion_final.csv')
stat1 = df1_n.describe()
#ajouter une colonne index afin de l'afficher sur le tableau stat generale
stat1['index'] = (['count', 'unique', 'top', 'freq'])
g1 = df1_n.groupby('Emotion').describe()
#ajouter une colonne mulitiindex afin de l'afficher dans le tableau stat generale par emotion
g1[('text','index')] = (['anger', 'fear', 'happy', 'love', 'sadness', 'surprise'])
# trier par ordre croissant la frequence des emotions 
g1 = g1.sort_values(by=[('Text','count')] , ascending = False)
df1 = df1_n[['Text','Emotion']]
df1.Emotion = df1_n.Emotion.replace(['happy','sadness','anger','fear','love','surprise'],[1,2,3,4,5,6])
# CountVectorizer
x1 = df1.Text
y1 = df1.Emotion
#stopword = stopwords.words('english')

vec1 = CountVectorizer()#, ngram_range=(2,2))
X1 = vec1.fit_transform(x1)
words1 = vec1.get_feature_names()
wsum = np.array(X1.sum(0))[0]
ix = wsum.argsort()[::-1]
wrank = wsum[ix] 
labels1 = [words1[i] for i in ix]

def subsample(x, step=350):
    return np.hstack((x[:30], x[10::step]))

freq1 = subsample(wrank)
r1 = np.arange(len(freq1))

def create_plot_bar(dataframe):
    trace = go.Bar(
                x = dataframe.index,
                y = dataframe[('Text','count')],
                name = "Frequence des Emotions",
                marker = dict(color = 'rgba(255, 87, 51, 0.5)',
                             line = dict(color ='rgb(0,0,0)',width =2.5)),
                text = dataframe[('Text','count')])

    layout = go.Layout(barmode = "group",
                  title = 'Fréquence d’apparition des Emotions ',
                  yaxis = dict(title = 'Emotion frequncy'),
                  xaxis = dict(title = 'Emotion rank'))
    data = [trace]#, layout = layout)
    return dcc.Graph(
                    id="plot2",
                    figure={
                        'data':data,
                        "layout":  go.Layout(#title = "Avez-vous vecu ou observé des conséquences négatives lié aux problèmes de santé mentale sur votre lieu de travail ?",
                            xaxis={'title': 'Emotion rank'},
                            yaxis={'title': 'Emotion frequency'})
                        }
                )
############################################## Analyse fichier 2 ###############################################
df_n = pd.read_csv('text_emotion.csv')
df_n = df_n[["content","sentiment"]]
stat = df_n.describe()
#ajouter une colonne index afin de l'afficher sur le tableau stat generale
stat['index'] = (['count', 'unique', 'top', 'freq'])
g = df_n.groupby('sentiment').describe()
g[('content','index')] = (['anger', 'boredom', 'empty', 'enthusiasm', 'fun', 'happiness', 'hate',
       'love', 'neutral', 'relief', 'sadness', 'surprise', 'worry'])#afin d'afficher l'indexsur le tableau das stat par emotion
# trier par ordre croissant la frequence des emotions 
g = g.sort_values(by=[('content','count')] , ascending = False)
df = df_n[["content","sentiment"]]
df.sentiment = df.sentiment.replace(['happiness','sadness','anger','worry','love','surprise','fun','relief'
                                     ,'empty','enthusiasm','boredom','hate','neutral']
                                    ,[1,2,3,4,5,6,7,8,9,10,11,12,13])
# CountVectorizer
x = df.content
y = df.sentiment
#stopword = stopwords.words('english')
vec = CountVectorizer()#, ngram_range=(2,2))
X = vec.fit_transform(x)
words = vec.get_feature_names()
wsum = np.array(X.sum(0))[0]
ix = wsum.argsort()[::-1]
wrank = wsum[ix] 
labels = [words[i] for i in ix]

def subsample(x, step=950):
    return np.hstack((x[:30], x[10::step]))
freq = subsample(wrank)
r = np.arange(len(freq))

def create_plot_bar1(dataframe):
    trace = go.Bar(
                x = dataframe.index,
                y = dataframe[('content','count')],
                name = "Frequence des Emotions",
                marker = dict(color = 'rgba(255, 87, 51, 0.5)',
                             line = dict(color ='rgb(0,0,0)',width =2.5)),
                text = dataframe[('content','count')])

    layout = go.Layout(barmode = "group",
                  title = 'Fréquence d’apparition des Emotions ',
                  yaxis = dict(title = 'Emotion frequncy'),
                  xaxis = dict(title = 'Emotion rank'))
    data = [trace]#, layout = layout)
    return dcc.Graph(
                    id="plot3",
                    figure={
                        'data':data,
                        "layout":  go.Layout(#title = "Avez-vous vecu ou observé des conséquences négatives lié aux problèmes de santé mentale sur votre lieu de travail ?",
                            xaxis={'title': 'Emotion rank'},
                            yaxis={'title': 'Emotion frequency'})
                        }
                )

####################################### Classification 1er fichier ###############################################################
ar1 = np.array([[0.949419,0.948276,0.908746,0.957343,0.964567,0.959064,0.935515,0.88888,0.857143,0.917094,0.910065,0.851973,0.926729,0.932606,0.915825,0.893451,0.800699,0.807339,0.978350,0.966216,0.915745,0.968436,0.984301,0.983193,0.965517,0.915423,0.835443]
               , [0.950678,0.959459,0.940039,0.968384,0.971657,0.961667,0.950606,0.913386,0.904494,0.922381,0.930801,0.904442,0.944876,0.944312,0.935216,0.917448,0.854812,0.879552,0.993107,0.993243,0.988558,0.995359,0.995640,0.992456,0.992424,0.979907,0.989011]
               , [0.998046,0.996971,0.994183,0.997504,0.998015,0.997498,0.997146,0.989114,0.997275,0.998207,0.996971,0.994037,0.997149,0.998413,0.997494,0.997146,0.989114,0.997275,0.998429,0.997670,0.995251,0.997860,0.998808,0.999167,0.997146,0.990683,0.997275]
              ,[0.998456,0.997437,0.994593,0.997504,0.998809,0.999166,0.997146,0.989114,0.997275,0.998588,0.997670,0.994871,0.997504,0.999206,1.000000,0.997146,0.989114,0.997275,0.998456,0.997437,0.994593,0.997504,0.998809,0.999166,0.997146,0.989114,0.997275]
              ,[0.866218,0.858807,0.774733,0.899000,0.902362,0.820208,0.792812,0.765125,0.650519,0.834925,0.807782,0.715609,0.833603,0.848904,0.780884,0.776119,0.650823,0.666667,0.862481,0.841333,0.767380,0.879535,0.879181,0.786084,0.806904,0.714019,0.733766]])
DFF1 = pd.DataFrame(ar1, index = ['Logistic_Regression', 'SVM_kernel=linear', 'SVM_kernel=poly','Decision_tree','KNN']
                      , columns = ['precision','score','recall','f1_score_happy','f1_score_sadness','f1_score_anger','f1_score_fear','f1_score_love','f1_score_surprise','precision_func','score_func','recall_func','f1_score_func_happy','f1_score_func_sadness','f1_score_func_anger','f1_score_func_fear','f1_score_func_love','f1_score_func_surprise'
                                  ,'precision_ngram','score_ngram','recall_ngram','f1_score_ngram_happy','f1_score_ngram_sadness','f1_score_ngram_anger','f1_score_ngram_fear','f1_score_ngram_love','f1_score_ngram_surprise'])
DFF1['index']=(['Logistic_Regression','SVM_kernel=linear','SVM_kernel=poly','Decision_tree','KNN'])
RL_simple = DFF1.loc['Logistic_Regression',['precision','score','recall','f1_score_happy'
                ,'f1_score_sadness','f1_score_anger','f1_score_fear'
                ,'f1_score_love','f1_score_surprise']]

RL_function = DFF1.loc['Logistic_Regression',['precision_func','score_func','recall_func','f1_score_func_happy'
                ,'f1_score_func_sadness','f1_score_func_anger','f1_score_func_fear'
                ,'f1_score_func_love','f1_score_func_surprise']]


RL_ngram = DFF1.loc['Logistic_Regression',['precision_ngram','score_ngram','recall_ngram','f1_score_ngram_happy'
                ,'f1_score_ngram_sadness','f1_score_ngram_anger','f1_score_ngram_fear'
                ,'f1_score_ngram_love','f1_score_ngram_surprise']]

SVM_L_simple = DFF1.loc['SVM_kernel=linear',['precision','score','recall','f1_score_happy'
                ,'f1_score_sadness','f1_score_anger','f1_score_fear'
                ,'f1_score_love','f1_score_surprise']]

SVM_L_function = DFF1.loc['SVM_kernel=linear',['precision_func','score_func','recall_func','f1_score_func_happy'
                ,'f1_score_func_sadness','f1_score_func_anger','f1_score_func_fear'
                ,'f1_score_func_love','f1_score_func_surprise']]

SVM_L_ngram =  DFF1.loc['SVM_kernel=linear',['precision_ngram','score_ngram','recall_ngram','f1_score_ngram_happy'
                ,'f1_score_ngram_sadness','f1_score_ngram_anger','f1_score_ngram_fear'
                ,'f1_score_ngram_love','f1_score_ngram_surprise']]

SVM_P_simple = DFF1.loc['SVM_kernel=poly',['precision','score','recall','f1_score_happy'
                ,'f1_score_sadness','f1_score_anger','f1_score_fear'
                ,'f1_score_love','f1_score_surprise']]

SVM_P_function = DFF1.loc['SVM_kernel=poly',['precision_func','score_func','recall_func','f1_score_func_happy'
                ,'f1_score_func_sadness','f1_score_func_anger','f1_score_func_fear'
                ,'f1_score_func_love','f1_score_func_surprise']]

SVM_P_ngram =  DFF1.loc['SVM_kernel=poly',['precision_ngram','score_ngram','recall_ngram','f1_score_ngram_happy'
                ,'f1_score_ngram_sadness','f1_score_ngram_anger','f1_score_ngram_fear'
                ,'f1_score_ngram_love','f1_score_ngram_surprise']]

Decision_Tree_simple = DFF1.loc['Decision_tree',['precision','score','recall','f1_score_happy'
                ,'f1_score_sadness','f1_score_anger','f1_score_fear'
                ,'f1_score_love','f1_score_surprise']]

Decision_Tree_function = DFF1.loc['Decision_tree',['precision_func','score_func','recall_func','f1_score_func_happy'
                ,'f1_score_func_sadness','f1_score_func_anger','f1_score_func_fear'
                ,'f1_score_func_love','f1_score_func_surprise']]

Decision_Tree_ngram =  DFF1.loc['Decision_tree',['precision_ngram','score_ngram','recall_ngram','f1_score_ngram_happy'
                                ,'f1_score_ngram_sadness','f1_score_ngram_anger','f1_score_ngram_fear'
                                ,'f1_score_ngram_love','f1_score_ngram_surprise']]

KNN_simple = DFF1.loc['KNN',['precision','score','recall','f1_score_happy'
                ,'f1_score_sadness','f1_score_anger','f1_score_fear'
                ,'f1_score_love','f1_score_surprise']]

KNN_function = DFF1.loc['KNN',['precision_func','score_func','recall_func','f1_score_func_happy'
                ,'f1_score_func_sadness','f1_score_func_anger','f1_score_func_fear'
                ,'f1_score_func_love','f1_score_func_surprise']]

KNN_ngram =  DFF1.loc['KNN',['precision_ngram','score_ngram','recall_ngram','f1_score_ngram_happy'
                ,'f1_score_ngram_sadness','f1_score_ngram_anger','f1_score_ngram_fear'
                ,'f1_score_ngram_love','f1_score_ngram_surprise']]
RL_general = pd.concat([RL_simple,RL_function, RL_ngram])
SVM_L_general = pd.concat([SVM_L_simple,SVM_L_function, SVM_L_ngram])
SVM_P_general = pd.concat([SVM_P_simple,SVM_P_function, SVM_P_ngram])
Decision_Tree_general = pd.concat([Decision_Tree_simple,Decision_Tree_function, Decision_Tree_ngram])
KNN_general = pd.concat([KNN_simple,KNN_function, KNN_ngram])

def create_plot_bar_class1(dataframe):
    trace1 = go.Bar(
                    x = ['precision','score','recall','f1_score_happy'
                    ,'f1_score_sadness','f1_score_anger','f1_score_fear'
                    ,'f1_score_love','f1_score_surprise'],
                    y = RL_general,
                    name = "Logistic Regression",
                    marker = dict(color = 'rgba(238, 235, 11 , 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = DFF1.iloc[0:1,1:9])
    trace2 = go.Bar(
                    x = ['precision','score','recall','f1_score_happy'
                    ,'f1_score_sadness','f1_score_anger','f1_score_fear'
                    ,'f1_score_love','f1_score_surprise'],
                    y = SVM_L_general,
                    name = "SVM kernel=linear",
                    marker = dict(color = 'rgba(245, 10, 27, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = DFF1.iloc[0:1,1:9])
    trace3 = go.Bar(
                    x = ['precision','score','recall','f1_score_happy'
                    ,'f1_score_sadness','f1_score_anger','f1_score_fear'
                    ,'f1_score_love','f1_score_surprise'],
                    y = SVM_P_general,
                    name = "SVM kernel=poly",
                    marker = dict(color = 'rgba(245, 152, 10, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = DFF1.iloc[0:1,1:9])
    trace4 = go.Bar(
                    x = ['precision','score','recall','f1_score_happy'
                    ,'f1_score_sadness','f1_score_anger','f1_score_fear'
                    ,'f1_score_love','f1_score_surprise'],
                    y = Decision_Tree_general,
                    name = "SDecision Tree",
                    marker = dict(color = 'rgba(64, 188, 11 , 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = DFF1.iloc[0:1,1:9])
    trace5 = go.Bar(
                    x = ['precision','score','recall','f1_score_happy'
                    ,'f1_score_sadness','f1_score_anger','f1_score_fear'
                    ,'f1_score_love','f1_score_surprise'],
                    y = KNN_general,
                    name = "KNN",
                    marker = dict(color = 'rgba(11, 99, 188, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = DFF1.iloc[0:1,1:9])
    data1 = [trace1, trace2, trace3, trace4, trace5]
    return dcc.Graph(
                    id="plot4",
                    figure={
                        'data':data1,
                        "layout":  go.Layout(#title = "Avez-vous vecu ou observé des conséquences négatives lié aux problèmes de santé mentale sur votre lieu de travail ?",
                            xaxis={'title': 'Différents Scores'},
                            yaxis={'title': 'Scores'})
                        }
                )

def create_plot_bar_class2(dataframe):
    trace1 = go.Bar(
                    x = ['Model_simple', 'Model + Function','Model + ngram'],
                    y = RL_general,
                    name = "Logistic Regression",
                    marker = dict(color = 'rgba(238, 235, 11 , 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = DFF1.iloc[0:1,1:9])
    trace2 = go.Bar(
                    x = ['Model_simple', 'Model + Function','Model + ngram'],
                    y = SVM_L_general,
                    name = "SVM kernel=linear",
                    marker = dict(color = 'rgba(245, 10, 27, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = DFF1.iloc[0:1,1:9])
    trace3 = go.Bar(
                    x = ['Model_simple', 'Model + Function','Model + ngram'],
                    y = SVM_P_general,
                    name = "SVM kernel=poly",
                    marker = dict(color = 'rgba(245, 152, 10, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = DFF1.iloc[0:1,1:9])
    trace4 = go.Bar(
                    x = ['Model_simple', 'Model + Function','Model + ngram'],
                    y = Decision_Tree_general,
                    name = "SDecision Tree",
                    marker = dict(color = 'rgba(64, 188, 11 , 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = DFF1.iloc[0:1,1:9])
    trace5 = go.Bar(
                    x = ['Model_simple', 'Model + Function','Model + ngram'],
                    y = KNN_general,
                    name = "KNN",
                    marker = dict(color = 'rgba(11, 99, 188, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = DFF1.iloc[0:1,1:9])
    data2 = [trace1, trace2, trace3, trace4, trace5]
    #layout = go.Layout(barmode = "group")
    #fig = go.Figure(data = data, layout = layout)
    return dcc.Graph(
                    id="plot4",
                    figure={
                        'data':data2,
                        "layout":  go.Layout(#title = "Avez-vous vecu ou observé des conséquences négatives lié aux problèmes de santé mentale sur votre lieu de travail ?",
                            xaxis={'title': 'Différentes Vectorisation'},
                            yaxis={'title': 'Scores'})
                        }
                )

f1_score_LR_simple_mean=(DFF1.loc['Logistic_Regression',['f1_score_happy','f1_score_sadness','f1_score_anger'
                                                        ,'f1_score_fear','f1_score_love','f1_score_surprise']]).mean()
f1_score_LR_function_mean= (DFF1.loc['Logistic_Regression',['f1_score_func_happy','f1_score_func_sadness'
                                                           ,'f1_score_func_anger','f1_score_func_fear'
                                                            ,'f1_score_func_love','f1_score_func_surprise']]).mean()
f1_score_LR_ngram_mean=(DFF1.loc['Logistic_Regression',['f1_score_ngram_happy','f1_score_ngram_sadness'
                                                       ,'f1_score_ngram_anger','f1_score_ngram_fear'
                                                        ,'f1_score_ngram_love','f1_score_ngram_surprise']]).mean()
f1_score_SVM_L_simple_mean=(DFF1.loc['SVM_kernel=linear',['f1_score_happy','f1_score_sadness','f1_score_anger'
                                                        ,'f1_score_fear','f1_score_love','f1_score_surprise']]).mean()
f1_score_SVM_L_func_mean=(DFF1.loc['SVM_kernel=linear',['f1_score_func_happy','f1_score_func_sadness'
                                                       ,'f1_score_func_anger','f1_score_func_fear'
                                                        ,'f1_score_func_love','f1_score_func_surprise']]).mean()
f1_score_SVM_L_ngram_mean=(DFF1.loc['SVM_kernel=linear',['f1_score_ngram_happy','f1_score_ngram_sadness'
                                                        ,'f1_score_ngram_anger','f1_score_ngram_fear'
                                                        ,'f1_score_ngram_love','f1_score_ngram_surprise']]).mean()

f1_score_SVM_P_simple_mean = (DFF1.loc['SVM_kernel=poly',['f1_score_happy','f1_score_sadness'
                                                         ,'f1_score_anger','f1_score_fear'
                                                        ,'f1_score_love','f1_score_surprise']]).mean()

f1_score_SVM_P_function_mean = (DFF1.loc['SVM_kernel=poly',['f1_score_func_happy','f1_score_func_sadness'
                                                           ,'f1_score_func_anger','f1_score_func_fear'
                                                            ,'f1_score_func_love','f1_score_func_surprise']]).mean()

f1_score_SVM_P_ngram_mean =  (DFF1.loc['SVM_kernel=poly',['f1_score_ngram_happy','f1_score_ngram_sadness'
                                                         ,'f1_score_ngram_anger','f1_score_ngram_fear'
                                                        ,'f1_score_ngram_love','f1_score_ngram_surprise']]).mean()

f1_score_Decision_Tree_simple_mean = (DFF1.loc['Decision_tree',['f1_score_happy','f1_score_sadness'
                                                               ,'f1_score_anger','f1_score_fear'
                                                                ,'f1_score_love','f1_score_surprise']]).mean()

f1_score_Decision_Tree_function_mean = (DFF1.loc['Decision_tree',['f1_score_func_happy','f1_score_func_sadness'
                                                                 ,'f1_score_func_anger','f1_score_func_fear'
                                                                ,'f1_score_func_love','f1_score_func_surprise']]).mean()

f1_score_Decision_Tree_ngram_mean =  (DFF1.loc['Decision_tree',['f1_score_ngram_happy','f1_score_ngram_sadness'
                                                               ,'f1_score_ngram_anger','f1_score_ngram_fear'
                                                            ,'f1_score_ngram_love','f1_score_ngram_surprise']]).mean()

f1_score_KNN_simple_mean = (DFF1.loc['KNN',['f1_score_happy','f1_score_sadness','f1_score_anger'
                                           ,'f1_score_fear','f1_score_love','f1_score_surprise']]).mean()

f1_score_KNN_function_mean = (DFF1.loc['KNN',['f1_score_func_happy','f1_score_func_sadness'
                                             ,'f1_score_func_anger','f1_score_func_fear'
                                            ,'f1_score_func_love','f1_score_func_surprise']]).mean()

f1_score_KNN_ngram_mean =  (DFF1.loc['KNN',['f1_score_ngram_happy','f1_score_ngram_sadness'
                                           ,'f1_score_ngram_anger','f1_score_ngram_fear'
                                        ,'f1_score_ngram_love','f1_score_ngram_surprise']]).mean()

ar_f1 = np.array([[f1_score_LR_simple_mean, f1_score_LR_function_mean, f1_score_LR_ngram_mean], [f1_score_SVM_L_simple_mean,f1_score_SVM_L_func_mean, f1_score_SVM_L_ngram_mean]
                  , [f1_score_SVM_P_simple_mean, f1_score_SVM_P_function_mean, f1_score_SVM_P_ngram_mean],[f1_score_Decision_Tree_simple_mean,f1_score_Decision_Tree_function_mean,f1_score_Decision_Tree_ngram_mean]
                 ,[f1_score_KNN_simple_mean,f1_score_KNN_function_mean,f1_score_KNN_ngram_mean]])
F1_score = pd.DataFrame(ar_f1, columns = ['f1_score_simple', 'f1_score_function', 'f1_score_ngram']
            , index = ['Logistic_Regression', 'SVM_kernel=linear', 'SVM_kernel=poly','Decision_tree','KNN'])

def create_plot_f1(dataframe):
    trace1 = go.Bar(
                    x = ['f1 score simple','f1 score function','f1 score ngram'],
                    y = F1_score.loc["Logistic_Regression"],
                    name = "Logistic_Regression",
                    marker = dict(color = 'rgba(238, 235, 11, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = F1_score.loc["Logistic_Regression"])
    trace2 = go.Bar(
                    x =['f1 score simple','f1 score function','f1 score ngram'],
                    y =F1_score.loc["SVM_kernel=linear"],
                    name = "SVM_kernel=linear",
                    marker = dict(color = 'rgba(245, 10, 27, 0.5)',
                                line = dict(color = 'rgb(0,0,0)',width = 1.5)),
                    text = F1_score.loc["SVM_kernel=linear"])

    trace3 = go.Bar(
                    x = ['f1 score simple','f1 score function','f1 score ngram'],
                    y = F1_score.loc["SVM_kernel=poly"],
                    name = "SVM_kernel=poly",
                    marker = dict(color = 'rgba(245, 152, 10, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = F1_score.loc["SVM_kernel=poly"])

    trace4 = go.Bar(
                    x = ['f1 score simple','f1 score function','f1 score ngram'],
                    y = F1_score.loc["Decision_tree"],
                    name = "Decision_tree",
                    marker = dict(color = 'rgba(64, 188, 11, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = F1_score.loc["Decision_tree"])

    trace5 = go.Bar(
                    x = ['f1 score simple','f1 score function','f1 score ngram'],
                    y = F1_score.loc["KNN"],
                    name = "KNN",
                    marker = dict(color = 'rgba(11, 99, 188, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = F1_score.loc["KNN"])
    data3 = [trace1, trace2, trace3, trace4,trace5]
    return dcc.Graph(
                    id="plot5",
                    figure={
                        'data':data3,
                        "layout":  go.Layout(#title = "Avez-vous vecu ou observé des conséquences négatives lié aux problèmes de santé mentale sur votre lieu de travail ?",
                            #xaxis={'title': 'Diffentes Vectorisation'},
                            yaxis={'title': 'Scores'})
                        }
                )

################################################# Classification de 2ème fichier###############################
ar_pipe_func = np.array([[16.108,0.229,0.355,0.177,0.177],[1.957,0.182,0.311,0.171,0.171],[437.085,0.2,0.282,0.114,0.099]
                     ,[25.490,0.161,0.278,0.156,0.156],[0.398,0.14,0.219,0.093,0.085]])
pipe_func = pd.DataFrame(ar_pipe_func, index = ['Logistic_Regression','SVM_kernel=linear','SVM_kernel=poly','Decision_tree','KNN']
                         ,columns = ['Time_func','Precision_func', 'Score_func', 'recall_func', 'f1_score_func'])
ar_pipe_simple = np.array([[19.751,0.222,0.354,0.175,0.173],[2.527,0.18,0.32,0.171,0.171],[683.706,0.213,0.291,0.116,0.096]
                     ,[26.867,0.17,0.28,0.158,0.161],[0.551,0.135,0.241,0.113,0.109]])
pipe_simple = pd.DataFrame(ar_pipe_simple, index = ['Logistic_Regression','SVM_kernel=linear','SVM_kernel=poly','Decision_tree','KNN']
                         ,columns = ['Time_simple','Precision_simple', 'Score_simple', 'recall_simple', 'f1_score_simple'])
ar_pipe_ngram = np.array([[23.958,0.222,0.354,0.175,0.173],[2.535,0.18,0.32,0.171,0.171],[674.944,0.213,0.291,0.116,0.096]
                     ,[26.786,0.164,0.279,0.155,0.157],[0.526,0.135,0.241,0.113,0.109]])
pipe_ngram = pd.DataFrame(ar_pipe_ngram, index = ['Logistic_Regression','SVM_kernel=linear','SVM_kernel=poly','Decision_tree','KNN']
                         ,columns = ['Time_ngram','Precision_ngram', 'Score_ngram', 'recall_ngram', 'f1_score_ngram'])
DFF = pd.concat([pipe_simple,pipe_func,pipe_ngram], axis =1)
DFF['index']=(['Logistic_Regression','SVM_kernel=linear','SVM_kernel=poly','Decision_tree','KNN'])
RL_simple1 = DFF.loc['Logistic_Regression'
                     ,['Precision_simple', 'Score_simple', 'recall_simple', 'f1_score_simple']]
RL_func1 = DFF.loc['Logistic_Regression'
                     ,['Precision_func', 'Score_func', 'recall_func', 'f1_score_func']]
RL_ngram1 = DFF.loc['Logistic_Regression'
                     ,['Precision_ngram', 'Score_ngram', 'recall_ngram', 'f1_score_ngram']]
SVM_L_simple1 = DFF.loc['SVM_kernel=linear'
                     ,['Precision_simple', 'Score_simple', 'recall_simple', 'f1_score_simple']]
SVM_L_func1 = DFF.loc['SVM_kernel=linear'
                     ,['Precision_func', 'Score_func', 'recall_func', 'f1_score_func']]
SVM_L_ngram1 = DFF.loc['SVM_kernel=linear'
                     ,['Precision_ngram', 'Score_ngram', 'recall_ngram', 'f1_score_ngram']]
SVM_P_simple1 = DFF.loc['SVM_kernel=poly'
                     ,['Precision_simple', 'Score_simple', 'recall_simple', 'f1_score_simple']]
SVM_P_func1 = DFF.loc['SVM_kernel=poly'
                     ,['Precision_func', 'Score_func', 'recall_func', 'f1_score_func']]
SVM_P_ngram1 = DFF.loc['SVM_kernel=poly'
                     ,['Precision_ngram', 'Score_ngram', 'recall_ngram', 'f1_score_ngram']]
Decision_Tree_simple1 = DFF.loc['Decision_tree'
                     ,['Precision_simple', 'Score_simple', 'recall_simple', 'f1_score_simple']]
Decision_Tree_func1 = DFF.loc['Decision_tree'
                     ,['Precision_func', 'Score_func', 'recall_func', 'f1_score_func']]
Decision_Tree_ngram1 = DFF.loc['Decision_tree'
                     ,['Precision_ngram', 'Score_ngram', 'recall_ngram', 'f1_score_ngram']]
KNN_simple1 = DFF.loc['KNN'
                     ,['Precision_simple', 'Score_simple', 'recall_simple', 'f1_score_simple']]
KNN_func1 = DFF.loc['KNN'
                     ,['Precision_func', 'Score_func', 'recall_func', 'f1_score_func']]
KNN_ngram1 = DFF.loc['KNN'
                     ,['Precision_ngram', 'Score_ngram', 'recall_ngram', 'f1_score_ngram']]
RL_general1 = pd.concat([RL_simple1,RL_func1, RL_ngram1])
SVM_L_general1 = pd.concat([SVM_L_simple1,SVM_L_func1, SVM_L_ngram1])
SVM_P_general1 = pd.concat([SVM_P_simple1,SVM_P_func1, SVM_P_ngram1])
Decision_Tree_general1 = pd.concat([Decision_Tree_simple1,Decision_Tree_func1, Decision_Tree_ngram1])
KNN_general1 = pd.concat([KNN_simple1,KNN_func1, KNN_ngram1])

def create_plot_class_2(dataframe):
    trace1 = go.Bar(
                    x = ['precision','score','recall','f1_score'],
                    y = RL_general1,
                    name = "Logistic Regression",
                    marker = dict(color = 'rgba(238, 235, 11 , 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = RL_general1)
    trace2 = go.Bar(
                    x = ['precision','score','recall','f1_score'],
                    y = SVM_L_general1,
                    name = "SVM kernel=linear",
                    marker = dict(color = 'rgba(245, 10, 27, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = SVM_L_general1)
    trace3 = go.Bar(
                    x = ['precision','score','recall','f1_score'],
                    y = SVM_P_general1,
                    name = "SVM kernel=poly",
                    marker = dict(color = 'rgba(245, 152, 10, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = SVM_P_general1)
    trace4 = go.Bar(
                    x = ['precision','score','recall','f1_score'],
                    y = Decision_Tree_general1,
                    name = "SDecision Tree",
                    marker = dict(color = 'rgba(64, 188, 11 , 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = Decision_Tree_general1,)
    trace5 = go.Bar(
                    x = ['precision','score','recall','f1_score'],
                    y = KNN_general1,
                    name = "KNN",
                    marker = dict(color = 'rgba(11, 99, 188, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = KNN_general1)
    data4 = [trace1, trace2, trace3, trace4, trace5]
    return dcc.Graph(
                    id="plot5",
                    figure={
                        'data':data4,
                        "layout":  go.Layout(#title = "Avez-vous vecu ou observé des conséquences négatives lié aux problèmes de santé mentale sur votre lieu de travail ?",
                            xaxis={'title': 'Diffents Scores'},
                            yaxis={'title': 'Scores'})
                        }
                )
def create_plot_class_2_1(dataframe):
    trace1 = go.Bar(
                    x = ['Model_simple', 'Model + Function','Model + ngram'],
                    y = RL_general1,
                    name = "Logistic Regression",
                    marker = dict(color = 'rgba(238, 235, 11 , 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = RL_general1)
    trace2 = go.Bar(
                    x = ['Model_simple', 'Model + Function','Model + ngram'],
                    y = SVM_L_general1,
                    name = "SVM kernel=linear",
                    marker = dict(color = 'rgba(245, 10, 27, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = SVM_L_general1)
    trace3 = go.Bar(
                    x = ['Model_simple', 'Model + Function','Model + ngram'],
                    y = SVM_P_general1,
                    name = "SVM kernel=poly",
                    marker = dict(color = 'rgba(245, 152, 10, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = SVM_P_general1)
    trace4 = go.Bar(
                    x = ['Model_simple', 'Model + Function','Model + ngram'],
                    y = Decision_Tree_general1,
                    name = "SDecision Tree",
                    marker = dict(color = 'rgba(64, 188, 11 , 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = Decision_Tree_general1)
    trace5 = go.Bar(
                    x = ['Model_simple', 'Model + Function','Model + ngram'],
                    y = KNN_general1,
                    name = "KNN",
                    marker = dict(color = 'rgba(11, 99, 188, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = KNN_general1)
    data5 = [trace1, trace2, trace3, trace4, trace5]
    return dcc.Graph(
                    id="plot6",
                    figure={
                        'data':data5,
                        "layout":  go.Layout(#title = "Avez-vous vecu ou observé des conséquences négatives lié aux problèmes de santé mentale sur votre lieu de travail ?",
                            xaxis={'title': 'Diffentes Vectorisations'},
                            yaxis={'title': 'Scores'})
                        }
                )
def create_plot_f1_2(dataframe):
    trace1 = go.Bar(
                    x = ['f1 score simple','f1 score function','f1 score ngram'],
                    y = DFF.loc['Logistic_Regression',['f1_score_simple','f1_score_func','f1_score_ngram']],
                    name = "Logistic_Regression",
                    marker = dict(color = 'rgba(238, 235, 11, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = DFF.loc['Logistic_Regression',['f1_score_simple','f1_score_func','f1_score_ngram']]) 
    trace2 = go.Bar(
                    x =['f1 score simple','f1 score function','f1 score ngram'],
                    y =DFF.loc["SVM_kernel=linear",['f1_score_simple','f1_score_func','f1_score_ngram']],
                    name = "SVM_kernel=linear",
                    marker = dict(color = 'rgba(245, 10, 27, 0.5)',
                                line = dict(color = 'rgb(0,0,0)',width = 1.5)),
                    text =DFF.loc[["SVM_kernel=linear"],['f1_score_simple','f1_score_func','f1_score_ngram']])
    trace3 = go.Bar(
                    x = ['f1 score simple','f1 score function','f1 score ngram'],
                    y = DFF.loc["SVM_kernel=poly",['f1_score_simple','f1_score_func','f1_score_ngram']],
                    name = "SVM_kernel=poly",
                    marker = dict(color = 'rgba(245, 152, 10, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text =DFF.loc[["SVM_kernel=poly"],['f1_score_simple','f1_score_func','f1_score_ngram']])
    trace4 = go.Bar(
                    x = ['f1 score simple','f1 score function','f1 score ngram'],
                    y = DFF.loc["Decision_tree",['f1_score_simple','f1_score_func','f1_score_ngram']],
                    name = "Decision_tree",
                    marker = dict(color = 'rgba(64, 188, 11, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = DFF.loc[["Decision_tree"],['f1_score_simple','f1_score_func','f1_score_ngram']])
    trace5 = go.Bar(
                    x = ['f1 score simple','f1 score function','f1 score ngram'],
                    y = DFF.loc["KNN",['f1_score_simple','f1_score_func','f1_score_ngram']],
                    name = "KNN",
                    marker = dict(color = 'rgba(11, 99, 188, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                    text = DFF.loc[["KNN"],['f1_score_simple','f1_score_func','f1_score_ngram']])
    data6 = [trace1, trace2, trace3, trace4,trace5]
    return dcc.Graph(
                    id="plot7",
                    figure={
                        'data':data6,
                        "layout":  go.Layout(#title = "Avez-vous vecu ou observé des conséquences négatives lié aux problèmes de santé mentale sur votre lieu de travail ?",
                            #xaxis={'title': 'Diffente'},
                            yaxis={'title': 'Scores'})
                        }
                )

################################################# data concat #################################################
ar_pipe_concat_ngram = np.array([[20.423,0.372,0.516,0.295,0.301],[3.203,0.336,0.507,0.312,0.315],[2561.826,0.323,0.425,0.192,0.182]
                                 ,[40.439,0.315,0.471,0.302,0.305],[0.776,0.337,0.352,0.205,0.212]])
pipe_concat_ngram = pd.DataFrame(ar_pipe_concat_ngram, index = ['Logistic_Regression','SVM_kernel=linear','SVM_kernel=poly','Decision_tree','KNN']
                         ,columns = ['Time_ngram','Precision_ngram', 'Score_ngram', 'recall_ngram', 'f1_score_ngram'])
ar_pipe_concat_simple = np.array([[24.954,0.372,0.516,0.295,0.301],[4.300,0.336,0.507,0.312,0.315],[2689.113,0.323,0.425,0.192,0.182]
                     ,[41.762,0.317,0.472,0.303,0.306],[0.838,0.337,0.352,0.205,0.212]])
pipe_concat_simple = pd.DataFrame(ar_pipe_concat_simple, index = ['Logistic_Regression','SVM_kernel=linear','SVM_kernel=poly','Decision_tree','KNN']
                         ,columns = ['Time_simple','Precision_simple', 'Score_simple', 'recall_simple', 'f1_score_simple'])
ar_pipe_concat_func = np.array([[25.655,0.372,0.516,0.295,0.301],[3.629,0.336,0.507,0.312,0.315],[2723.285,0.323,0.425,0.192,0.182]
                     ,[40.556,0.313,0.468,0.3,0.304],[0.798,0.337,0.352,0.205,0.212]])
pipe_concat_func = pd.DataFrame(ar_pipe_concat_func, index = ['Logistic_Regression','SVM_kernel=linear','SVM_kernel=poly','Decision_tree','KNN']
                         ,columns = ['Time_func','Precision_func', 'Score_func', 'recall_func', 'f1_score_func'])
DFF_concat = pd.concat([pipe_concat_simple,pipe_concat_func,pipe_concat_ngram], axis =1)
DFF_concat['index']=(['Logistic_Regression','SVM_kernel=linear','SVM_kernel=poly','Decision_tree','KNN'])
RL_concat_simple = DFF_concat.loc['Logistic_Regression'
                     ,['Precision_simple', 'Score_simple', 'recall_simple', 'f1_score_simple']]
RL_concat_func = DFF_concat.loc['Logistic_Regression'
                     ,['Precision_func', 'Score_func', 'recall_func', 'f1_score_func']]
RL_concat_ngram = DFF_concat.loc['Logistic_Regression'
                     ,['Precision_ngram', 'Score_ngram', 'recall_ngram', 'f1_score_ngram']]
SVM_L_concat_simple = DFF_concat.loc['SVM_kernel=linear'
                     ,['Precision_simple', 'Score_simple', 'recall_simple', 'f1_score_simple']]
SVM_L_concat_func = DFF_concat.loc['SVM_kernel=linear'
                     ,['Precision_func', 'Score_func', 'recall_func', 'f1_score_func']]
SVM_L_concat_ngram = DFF_concat.loc['SVM_kernel=linear'
                     ,['Precision_ngram', 'Score_ngram', 'recall_ngram', 'f1_score_ngram']]
SVM_P_concat_simple = DFF_concat.loc['SVM_kernel=poly'
                     ,['Precision_simple', 'Score_simple', 'recall_simple', 'f1_score_simple']]
SVM_P_concat_func = DFF_concat.loc['SVM_kernel=poly'
                     ,['Precision_func', 'Score_func', 'recall_func', 'f1_score_func']]
SVM_P_concat_ngram = DFF_concat.loc['SVM_kernel=poly'
                     ,['Precision_ngram', 'Score_ngram', 'recall_ngram', 'f1_score_ngram']]
Decision_Tree_concat_simple = DFF_concat.loc['Decision_tree'
                     ,['Precision_simple', 'Score_simple', 'recall_simple', 'f1_score_simple']]
Decision_Tree_concat_func = DFF_concat.loc['Decision_tree'
                     ,['Precision_func', 'Score_func', 'recall_func', 'f1_score_func']]
Decision_Tree_concat_ngram = DFF_concat.loc['Decision_tree'
                     ,['Precision_ngram', 'Score_ngram', 'recall_ngram', 'f1_score_ngram']]
KNN_concat_simple = DFF_concat.loc['KNN'
                     ,['Precision_simple', 'Score_simple', 'recall_simple', 'f1_score_simple']]
KNN_concat_func = DFF_concat.loc['KNN'
                     ,['Precision_func', 'Score_func', 'recall_func', 'f1_score_func']]
KNN_concat_ngram = DFF_concat.loc['KNN'
                     ,['Precision_ngram', 'Score_ngram', 'recall_ngram', 'f1_score_ngram']]
RL_concat_general = pd.concat([RL_concat_simple,RL_concat_func, RL_concat_ngram])
SVM_L_concat_general = pd.concat([SVM_L_concat_simple,SVM_L_concat_func, SVM_L_concat_ngram])
SVM_P_concat_general = pd.concat([SVM_P_concat_simple,SVM_P_concat_func, SVM_P_concat_ngram])
Decision_Tree_concat_general = pd.concat([Decision_Tree_concat_simple,Decision_Tree_concat_func, Decision_Tree_concat_ngram])
KNN_concat_general = pd.concat([KNN_concat_simple,KNN_concat_func, KNN_concat_ngram])

def create_plot_bar_concat(dataframe):
    trace1 = go.Bar(
                x = ['precision','score','recall','f1_score'],
                y = RL_concat_general,
                name = "Logistic Regression",
                marker = dict(color = 'rgba(238, 235, 11 , 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                text = RL_concat_general)
    trace2 = go.Bar(
                x = ['precision','score','recall','f1_score'],
                y = SVM_L_concat_general,
                name = "SVM kernel=linear",
                marker = dict(color = 'rgba(245, 10, 27, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                text = SVM_L_concat_general)
    trace3 = go.Bar(
                x = ['precision','score','recall','f1_score'],
                y = SVM_P_concat_general,
                name = "SVM kernel=poly",
                marker = dict(color = 'rgba(245, 152, 10, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                text = SVM_P_concat_general)
    trace4 = go.Bar(
                x = ['precision','score','recall','f1_score'],
                y = Decision_Tree_concat_general,
                name = "SDecision Tree",
                marker = dict(color = 'rgba(64, 188, 11 , 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                text = Decision_Tree_concat_general)
    trace5 = go.Bar(
                x = ['precision','score','recall','f1_score'],
                y = KNN_concat_general,
                name = "KNN",
                marker = dict(color = 'rgba(11, 99, 188, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                text = KNN_concat_general)
    data7 = [trace1, trace2, trace3, trace4, trace5]
    return dcc.Graph(
                    id="plot8",
                    figure={
                        'data':data7,
                        "layout":  go.Layout(#title = "Avez-vous vecu ou observé des conséquences négatives lié aux problèmes de santé mentale sur votre lieu de travail ?",
                            xaxis={'title': 'Diffents Scores'},
                            yaxis={'title': 'Scores'})
                        }
                )

def create_plot_bar_concat2(dataframe):
    trace1 = go.Bar(
                x = ['Model_simple', 'Model + Function','Model + ngram'],
                y = RL_concat_general,
                name = "Logistic Regression",
                marker = dict(color = 'rgba(238, 235, 11 , 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                text = RL_concat_general)
    trace2 = go.Bar(
                x = ['Model_simple', 'Model + Function','Model + ngram'],
                y = SVM_L_concat_general,
                name = "SVM kernel=linear",
                marker = dict(color = 'rgba(245, 10, 27, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                text = SVM_L_concat_general)
    trace3 = go.Bar(
                x = ['Model_simple', 'Model + Function','Model + ngram'],
                y = SVM_P_concat_general,
                name = "SVM kernel=poly",
                marker = dict(color = 'rgba(245, 152, 10, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                text = SVM_P_concat_general)
    trace4 = go.Bar(
                x = ['Model_simple', 'Model + Function','Model + ngram'],
                y = Decision_Tree_concat_general,
                name = "SDecision Tree",
                marker = dict(color = 'rgba(64, 188, 11 , 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                text = Decision_Tree_concat_general)
    trace5 = go.Bar(
                x = ['Model_simple', 'Model + Function','Model + ngram'],
                y = KNN_concat_general,
                name = "KNN",
                marker = dict(color = 'rgba(11, 99, 188, 0.5)',
                                line = dict(color ='rgb(0,0,0)',width =1.5)),
                text = KNN_concat_general)
    data8 = [trace1, trace2, trace3, trace4, trace5]
    return dcc.Graph(
                    id="plot9",
                    figure={
                        'data':data8,
                        "layout":  go.Layout(#title = "Avez-vous vecu ou observé des conséquences négatives lié aux problèmes de santé mentale sur votre lieu de travail ?",
                            xaxis={'title': 'Diffentes Vectorisations'},
                            yaxis={'title': 'Scores'})
                        }
                )
def create_plot_bar_concat3(dataframe):
    trace1 = go.Bar(
                x = ['f1 score simple','f1 score function','f1 score ngram'],
                y = DFF_concat.loc['Logistic_Regression',['f1_score_simple','f1_score_func','f1_score_ngram']],
                name = "Logistic_Regression",
                marker = dict(color = 'rgba(238, 235, 11, 0.5)',
                            line = dict(color ='rgb(0,0,0)',width =1.5)),
                text = DFF_concat.loc['Logistic_Regression',['f1_score_simple','f1_score_func','f1_score_ngram']]) 
    trace2 = go.Bar(
                x =['f1 score simple','f1 score function','f1 score ngram'],
                y =DFF_concat.loc["SVM_kernel=linear",['f1_score_simple','f1_score_func','f1_score_ngram']],
                name = "SVM_kernel=linear",
                marker = dict(color = 'rgba(245, 10, 27, 0.5)',
                            line = dict(color = 'rgb(0,0,0)',width = 1.5)),
                text =DFF_concat.loc[["SVM_kernel=linear"],['f1_score_simple','f1_score_func','f1_score_ngram']])
    trace3 = go.Bar(
                x = ['f1 score simple','f1 score function','f1 score ngram'],
                y = DFF_concat.loc["SVM_kernel=poly",['f1_score_simple','f1_score_func','f1_score_ngram']],
                name = "SVM_kernel=poly",
                marker = dict(color = 'rgba(245, 152, 10, 0.5)',
                            line = dict(color ='rgb(0,0,0)',width =1.5)),
                text =DFF_concat.loc[["SVM_kernel=poly"],['f1_score_simple','f1_score_func','f1_score_ngram']])
    trace4 = go.Bar(
                x = ['f1 score simple','f1 score function','f1 score ngram'],
                y = DFF_concat.loc["Decision_tree",['f1_score_simple','f1_score_func','f1_score_ngram']],
                name = "Decision_tree",
                marker = dict(color = 'rgba(64, 188, 11, 0.5)',
                            line = dict(color ='rgb(0,0,0)',width =1.5)),
                text = DFF_concat.loc[["Decision_tree"],['f1_score_simple','f1_score_func','f1_score_ngram']])
    trace5 = go.Bar(
                x = ['f1 score simple','f1 score function','f1 score ngram'],
                y = DFF_concat.loc["KNN",['f1_score_simple','f1_score_func','f1_score_ngram']],
                name = "KNN",
                marker = dict(color = 'rgba(11, 99, 188, 0.5)',
                            line = dict(color ='rgb(0,0,0)',width =1.5)),
                text = DFF_concat.loc[["KNN"],['f1_score_simple','f1_score_func','f1_score_ngram']])
    data9 = [trace1, trace2, trace3, trace4,trace5]
    return dcc.Graph(
                    id="plot10",
                    figure={
                        'data':data9,
                        "layout":  go.Layout(#title = "Avez-vous vecu ou observé des conséquences négatives lié aux problèmes de santé mentale sur votre lieu de travail ?",
                            #xaxis={'title': 'Diffentes Vectorisations'},
                            yaxis={'title': 'Scores'})
                        }
                )
################################ essayer d'améliorer la prédiction ############################
ar_am = ([[22.314,0.372,0.516,0.295,0.301],[4.204,0.336,0.507,0.312,0.315],[2664.365,0.323,0.425,0.192,0.182]
          ,[41.091,0.316,0.472,0.303,0.306],[0.841,0.337,0.352,0.205,0.212]])
DFF_am = pd.DataFrame(ar_am, index = ['Logistic_Regression','SVM_kernel=linear','SVM_kernel=poly','Decision_tree','KNN']
                        ,columns = ['Time','Precision', 'Score', 'recall', 'f1_score'])
DFF_am['index']=(['Logistic_Regression','SVM_kernel=linear','SVM_kernel=poly','Decision_tree','KNN'])

def get_menu():
    menu = html.Div(style={'font-size': '20px', 'background-color':'#EBF0F5', 'textAlign': 'center', 'color':'rgba(0,0,7,0.7)'}, children=[ 

        dcc.Link('Analyse et Traitement des données|', href='/AnalyseTraitementdesdonnées', className='mb-3'),

        dcc.Link('Résultats des Classifications', href='/ResultatsdesClassifications', className="mb-3")
    ], className="rows")
    return menu
    
def generate_table3(dataframe):
    return dash_table.DataTable(
    data=dataframe.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in dataframe.columns],
    css=[{'selector': '.dash-cell div.dash-cell-value', 'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'}],
    style_header={
                'backgroundColor': '#435684',
                'color' : 'white',
                'fontWeight': 'bold'
            },
    style_cell = {"fontFamily": "Arial", "size": 15, 'textAlign': 'center'},
    style_table={'overflowX': 'scroll',
                         'maxHeight': '1000px',
                         'maxWidth': '100%',
                         'overflowY': 'scroll',
                         'maxHeight': '300px',
                         'maxWidth': '1500px',
                         'width': '100%',
                         'Height' : '49%',
                         'display': 'inline-block',
                         'vertical-align': 'middle'},
            style_cell_conditional=[
                {
                    'if': {'column_id': c},
                    'textAlign': 'left'
                } for c in ['Date', 'Region']
            ],
    page_size=10
    )
def generate_table(dataframe, max_rows=15):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

def generate_table1(dataframe):
    return html.Div([   
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in dataframe.columns],
            data=dataframe.to_dict('records'),
            editable=True,
            css=[{'selector': '.dash-cell div.dash-cell-value', 'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'}],
            style_table={'overflowX': 'scroll',
                         'maxHeight': '1000px',
                         'maxWidth': '100%',
                         'overflowY': 'scroll',
                         'maxHeight': '300px',
                         'maxWidth': '1500px',
                         'width': '100%',
                         'Height' : '49%',
                         'display': 'inline-block',
                         'vertical-align': 'middle'},
            style_cell = {"fontFamily": "Arial", "size": 10, 'textAlign': 'center'},
            style_cell_conditional=[
                {
                    'if': {'column_id': c},
                    'textAlign': 'left'
                } for c in ['Date', 'Region']
            ],
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#F2A2AB',
                    'color': 'white'
                }
            ],
            style_header={
                'backgroundColor': '#C33544',
                'color' : 'white',
                'fontWeight': 'bold'
            },
        )
    ],className="twelve columns")
# Bar charts (words)1
fig = go.Figure(data=[
    go.Bar(name='frequence des mots', x=r1, y=freq1, marker = dict(color = 'rgba(255, 87, 51, 0.5)',
                             line = dict(color ='rgb(0,0,0)',width =2.5)),
                text = subsample(labels1))
])
fig.update_layout(barmode='group',title = 'Fréquence d’apparition des mots ',
                  yaxis = dict(title = 'word frequncy'),
                  xaxis = dict(title = 'word rank'))                  
fig.update_xaxes(
        tickmode='array',
        tickvals = r1,
        ticktext = labels1
)

# Bar charts (words)2
fig2 = go.Figure(data=[
    go.Bar(name='frequence des mots', x=r, y=freq, marker = dict(color = 'rgba(255, 87, 51, 0.5)',
                             line = dict(color ='rgb(0,0,0)',width =2.5)),
                text = subsample(labels))
])
fig2.update_layout(barmode='group',title = 'Fréquence d’apparition des mots ',
                  yaxis = dict(title = 'word frequncy'),
                  xaxis = dict(title = 'word rank'))
fig2.update_xaxes(
        tickmode='array',
        tickvals = r,
        ticktext = labels
)

text = '''
On commence à analyser les données du premier fichier "Emotion_Final.csv" : 
- Les statistiques descriptives générales
- Les statistiques descriptives par rapport aux emotions
''' 
text1 = '''
J'ai changer les Emotions en chiffres :
- happy = 1
- sadness = 2
- anger = 3
- fear = 4
- love = 5
- surprise = 6
'''
text2 = '''
- Pour le tableau des statistiques descriptives générales on remarque :
    - Il y a 6 emotions dans ce dataframe
    - L'emotion la plus fréquente est "happy"

- Pour le tableau des statistiques descriptives par rapport aux emotions on obtient :
    - Toutes les emotions triées par ordre décroissant(de la plus fréquente "happy" à la moins fréquente "surprise")
'''
text3 = '''
Cet histogramme represente la fréquence des mots c'est à dire le nombre de fois d'apparition pour chaque mot :
- Les 5 mots les plus fréquents sans utiliser le stopwords sont :
    - feel
    - and
    - to
    - the
    - of
- Les 5 mots les plus fréquents en utilisant le stopwords sont :
    - feel
    - feeling
    - like
    - im
    - really
'''
text4 = '''
Cet histogramme represente la fréquence des emotions c'est à dire le nombre de fois d'apparition pour chaque emotion :
- Les emotions classées par ordre décroissant :
    - happy
    - sadness
    - anger
    - fear
    - love
    - surprise
'''
text21='''
On commence à analyser les données du deuxième fichier "text_emotion.csv" : 
- Les statistiques descriptives générales
- Les statistiques descriptives par rapport aux emotions
'''
text25='''
- Pour le tableau des statistiques descriptives générales on remarque :
    - Il y a 13 emotions dans ce dataframe
    - L'emotion la plus fréquente est "neutral"

- Pour le tableau des statistiques descriptives par rapport aux emotions on obtient :
    - Toutes les emotions triées par ordre décroissant(de la plus fréquente "neutral" à la moins fréquente "anger")
'''
text5 = '''
J'ai changer les Emotions en chiffres :
- happy = 1
- sadness = 2
- anger = 3
- fear = 4
- love = 5
- surprise = 6
- fun = 7
- relief = 8
- empty = 9
- enthusiasm = 10
- boredom = 11
- hate = 12
- neutral = 13
'''
text6 = '''
Cet histogramme represente la fréquence des mots c'est à dire le nombre de fois d'apparition pour chaque mot :
- Les 5 mots les plus fréquents sans utiliser le stopwords sont :
    - to
    - the
    - my
    - you
    - it
- Les 5 mots les plus fréquents en utilisant le stopwords sont :
    - day
    - good
    - get
    - like
    - quot
'''
text7 = '''
Cet histogramme represente la fréquence des emotions c'est à dire le nombre de fois d'apparition pour chaque emotion :
- Les emotions classées par ordre décroissant :
    - neutral
    - worry
    - happiness
    - sadness
    - love
    - surprise
    - fun
    - relief
    - hate
    -empty
    - enthusiasm
    - boredom
    - anger
'''
text8 = '''
Pour les premières données je n'ai pas utilisé des pipelines, j'ai utilisé 5 classifications independamment les unes des autres : 
- Regression Logistique
- SVM avec un kernel = linear
- SVM avec un kernel = poly
- Decision Tree 
- KNN 

Pour ces 5 classification j'ai utislisé TfidfVectorizer pour la vectorisation des données :  
    - Dans un premier temps TfidfVectorizer avec stopwords.  
    - Dans un deuxième temps TfidfVectorizer avec stopwords et une fonction pour la lemmatisation et le stemming  
    - Dans un troisième temps TfidfVectorizer avec stopwords et ngram_range = (1,2) 

Puis j'ai concaténé tous les résultats des classifications dans un seul tableau
'''
text9 = '''
Cet histogramme represente les Scores par classification regroupé par type de score.  
On remarque que 2 classifications se détachent SVM(kernel=poly) et Decision Tree  
Pour obtenir le f1_score pour chaque emotion il faut mettre l'argument average=None,le score F1 peut être interprété comme une moyenne pondérée de la précision et du rappel, où un score F1 atteint sa meilleure valeur à 1 et son pire score à 0. La contribution relative de la précision et du rappel au score F1 est égale.  
La formule du score F1 est:  
F1 = 2 * (precision * recall) / (precision + recall)  
Dans le cas multi-classes et multi-étiquettes, il s'agit de la moyenne du score F1 de chaque classe  
Je calcule la moyenne des f1_score pour chaque classification
'''
text10 = '''
Cet histogramme représente la comparaision des classifications par rapport aux 3 types des vectorisation des données :  
- stopwords
- stopwords + fonction de lemmatisation et stemming
- stopwords + ngram_range = (1,2) 

On remarque que la vectorisation just avec le stopwords et légèrement plus performante  
'''
text11 = '''
Comme j'ai des f1_score par emotion, je calcul la moyenne des f1_score par score et par classification.  
Afin de pouvoir faire une comparison des classifications par rapport aux f1_score
'''
text12 = '''
On remarque dans cet histograme montre qu'il y a 2 classifications plus performates que les autres c'est toujours le SVM(kernel=poly) et Decision Tree, mais en utilisant largument nrgam_range = (1,2), les performances des autres classifications s'améliorent.
'''
text13 ='''
Pour les deuxième données j'ai utilisé des pipelines, j'ai utilisé 5 classifications : 
- Regression Logistique
- SVM avec un kernel = linear
- SVM avec un kernel = poly
- Decision Tree 
- KNN 

Pour ces 5 classification j'ai utislisé TfidfVectorizer pour la vectorisation des données :  
    - Dans un premier temps TfidfVectorizer avec stopwords.  
    - Dans un deuxième temps TfidfVectorizer avec stopwords et une fonction pour la lemmatisation et le stemming  
    - Dans un troisième temps TfidfVectorizer avec stopwords et ngram_range = (1,2) 

Puis j'ai concaténé tous les résultats des pipelines dans un seul tableau
'''
text14='''
On obtient des résultats vraiment bas, même si sur cet histogramme le score de la Regression Logistique est un peu plus élevé par rapport aux autres classifications
'''
text15='''
Dans cet histogramme on peu dire que dans cette situation la vectorisation la plus performante et celle avec le stopwords et la fonction de lemmatisation et le stemming
'''
text16 = '''
Dans cet histogramme on remarque que le f1_score est plus élevé pour les classifications :
- Regression Logistique 
- SVM(kernel = linear)
- Decision Tree
'''
text17 = '''
Ici on va concaténer les 2 jeux de données.  
Pour cela je vais regrouper les emotions les plus proches et je les ai numérotées :
- happy et happiness = 1            - func = 7
- sadness = 2
- anger = 3
- worry et fear = 4
- love = 5
- surprise = 6
- func = 7
- relief = 8
- empty = 9
- enthusiasm = 10
- boredom = 11
- hate = 12
- neutral = 13
'''
text22='''
On remarque que les résulats des pipelines sont assez similaires
'''
text18 = '''
On remarque que le score est assez élevé pour 3 types de classifications :  
- Regression Logistic
- SVM(kernel = linear)
- Decision Tree
'''
text19= '''
Cet histogramme nous montre que la vectorisation qui produit les scores les plus élevé est :  
- La vectorisation avec stopwords et la fonction de lemmatisation et stemming
'''
text20='''
On remarque que les classifications qui ont le f1_score le plus élevé sont :  
- Regression Logistique 
- SVM(kernel=linear)
- Decision Tree 
'''
text26='''
On remarque que même en ajoutant des arguments comme :
- sublinear_tf = True, 
- min_df = 5,
- norm = l2  

en plus du stopwords et le ngram_range = (1,2) 
 
on obtient à peu près les mêmes scores, les plus performants sont :
- Logistic Regression
- SVM linear
'''

layout1 = html.Div([
    get_menu(),
    dcc.Tabs([
        dcc.Tab(label="Analyse du fichier Emotion_Final.csv",style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}, children=[
            #html.H2('Analyse et raitement des données du premier fichier Emotion_Final.csv', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            html.Div(dcc.Markdown(text, style={'font-size':'18px'})),
            html.H2(children='Tableau des données',style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            generate_table3(df1_n),
            # Download Button
                    html.Div([
                        html.A(html.Button('Download data'),
                            id="download-button",
                            download='Emotion_final.csv',
                            href="data:text/csv;charset=utf-8,"+quote(df1_n.to_csv(index=False)),
                            target="_blank"),
                        ]),
            html.H2(children='Les statistiques descriptives générales', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            generate_table3(stat1),
            html.Br(),
            html.H2(children='Les statistiques descriptives par rapport aux emotions',style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            generate_table(g1),
            html.Br(),
            html.Br(),
            html.H2(children="L'analyse des 2 tableaux de statistiques"),
            html.Div(dcc.Markdown(text2, style={'font-size':'18px'})),
            html.Br(),
            html.H2(children='Tableau des données traités',style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            generate_table3(df1),
            html.Div(dcc.Markdown(text1, style={'font-size':'18px'})),
            html.H2(children='La frequence des mots sans le stopwords', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            dcc.Graph(
                id = 'id1',
                figure=fig
            ),
            html.Br(),
            html.H2(children='La frequence des mots avec le stopwords', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            html.Img(src=app.get_asset_url('image1.png'), height=400, width=1000, style={'padding-left':'7vw'}),
            html.Div(dcc.Markdown(text3, style={'font-size':'18px'})),
            html.H2(children='La frequence des Emotion', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            create_plot_bar(g1),
            html.Div(dcc.Markdown(text4, style={'font-size':'18px'})) 
        ]),
        dcc.Tab(label="Analyse du fichier tex_emeotion.csv", children=[
            #html.H2('Analyse et raitement des données du deuxième fichier text_emotion.csv', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            html.Div(dcc.Markdown(text21, style={'font-size':'18px'})),
            html.H2(children='Tableau des données',style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            generate_table3(df_n),
            # Download Button
                    html.Div([
                        html.A(html.Button('Download data'),
                            id="download-button",
                            download='text_emotion.csv',
                            href="data:text/csv;charset=utf-8,"+quote(df_n.to_csv(index=False)),
                            target="_blank"),
                        ]),
            html.H2(children='Les statistiques descriptives générales', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            generate_table3(stat),
            html.Br(),
            html.H2(children='Les statistiques descriptives par rapport aux emotions',style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            generate_table(g),
            html.Br(),
            html.Br(),
            html.H2(children="L'analyse des 2 tableaux de statistiques"),
            html.Div(dcc.Markdown(text25, style={'font-size':'18px'})),
            html.Br(),
            html.H2(children='Tableau des données traités',style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            generate_table3(df),
            html.Div(dcc.Markdown(text5, style={'font-size':'18px'})),
            html.H2(children='La frequence des mots sans le stopwords', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            dcc.Graph(
                id = 'id1',
                figure=fig2
            ),
            html.Br(),
            html.H2(children='La frequence des mots avec le stopwords', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            html.Img(src=app.get_asset_url('image3.png'), height=400, width=1000, style={'padding-left':'7vw'}),
            html.Div(dcc.Markdown(text6, style={'font-size':'18px'})),
            html.H2(children='La frequence des Emotion', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            create_plot_bar1(g),
            html.Div(dcc.Markdown(text7, style={'font-size':'18px'}))

        ]),
        
    ]),
    html.Div(id='app-1-display-value'),
    dcc.Link('Go to Resultats des Classifications', href='/ResultatsdesClassifications')
])

layout2 = html.Div([
    #html.H1('Analyse des Emotions', style={'textAlign': 'center', 'color':'rgba(35, 32, 37 ,1.7)', 'border':'3px double black'}),
    #html.H2('Résultas des Classifieurs', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
    get_menu(),
    dcc.Tabs([
        dcc.Tab(label="Résultas des Classifieurs du fichier Emotion_Final.csv", children=[
            #html.H2(children='Classifications des premières données Emotion_final.csv',style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            html.Div(dcc.Markdown(text8, style={'font-size':'18px'})),
            html.H2(children='Tableau recapitulatif des resultats des 5 classifications',style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            generate_table3(DFF1),
            html.H2(children='Comparaison des Classifications par rapport aux Scores(précision,score,recall,f1_score)', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            create_plot_bar_class1(DFF1),
            html.Div(dcc.Markdown(text9, style={'font-size':'18px'})),
            html.H2(children='Comparaison des Classifications par rapport aux différentes Vectorisation(simple,avec fonction,avec ngram)', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            create_plot_bar_class2(DFF1),
            html.Div(dcc.Markdown(text10, style={'font-size':'18px'})),
            html.H2('Tableau des moyennes f1_score', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            generate_table3(F1_score),
            html.Br(),
            html.Div(dcc.Markdown(text11, style={'font-size':'18px'})),
            html.Br(),
            html.H2('Comparaison des Classifications par rapport aux moyennes f1_score', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            create_plot_f1(F1_score),
            html.Div(dcc.Markdown(text12, style={'font-size':'18px'}))
        ]),
        dcc.Tab(label="Résultas des Classifieurs du fichier text_emotion.csv", children=[
            html.Div(dcc.Markdown(text13, style={'font-size':'18px'})),
            html.H2('Tableau Pipeline(lemmatisation, stemming, stopwords, ngrame_range=(1,2))', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            generate_table3(DFF),
            html.Br(),
            html.H2(children='Comparaison des Classifications par rapport aux Scores(précision,score,recall,f1_score)', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            create_plot_class_2(DFF),
            html.Div(dcc.Markdown(text14, style={'font-size':'18px'})),
            html.H2(children='Comparaison des Classifications par rapport aux différentes Vectorisation(simple,avec fonction,avec ngram)', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            create_plot_class_2_1(DFF),
            html.Div(dcc.Markdown(text15, style={'font-size':'18px'})),
            html.Br(),
            html.H2('Comparaison des Classifications par rapport aux f1_score', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            create_plot_f1_2(DFF),
            html.Div(dcc.Markdown(text16, style={'font-size':'18px'}))
        ]),
        dcc.Tab(label="Données concaténées", children=[
            html.Div(dcc.Markdown(text17, style={'font-size':'18px'})),
            html.H2('Tableau recapitulatif des 3 Pipelines des données concaténées', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            generate_table3(DFF_concat),
            html.Br(),
            html.Div(dcc.Markdown(text22, style={'font-size':'18px'})),
            html.Br(),
            html.H2(children='Comparaison des Classifications par rapport aux Scores(précision,score,recall,f1_score)', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            create_plot_bar_concat(DFF_concat),
            html.Br(),
            html.Div(dcc.Markdown(text18, style={'font-size':'18px'})),
            html.H2(children='Comparaison des Classifications par rapport aux différentes Vectorisation(simple,avec fonction,avec ngram)', style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            create_plot_bar_concat2(DFF_concat),
            html.Br(),
            html.Div(dcc.Markdown(text19, style={'font-size':'18px'})),
            create_plot_bar_concat3(DFF_concat),
            html.Div(dcc.Markdown(text20, style={'font-size':'18px'})),
            html.Br(),
            html.Br(),
            html.H2(children="Test pour améliorer les scores",style={'textAlign':'center', 'color':'rgba(35, 32, 37 ,1.7'}),
            generate_table3(DFF_am),
            html.Br(),
            html.Div(dcc.Markdown(text26, style={'font-size':'18px'}))

        ]),
    ])
    ,
    html.Div(id='app-2-display-value'),
    dcc.Link('Go to Analyse et Traitement des données', href='/AnalyseTraitementdesdonnées')
])