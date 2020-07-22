# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:54:30 2016

@author: chenxi.zhang
"""

# STEP ONE
# load the data
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
from sqlalchemy import create_engine
engine_subordinate = create_engine('mysql+mysqldb://dbaread:8TvWzIH0n&yiqg3g@localhost:3308/yhouse', connect_args={'charset':'utf8'})

import pandas as pd
purchase_no = pd.read_sql_query("""select a.meal_info_id, sum(1/a.day_diff) purchase_cnt, sum(a.pay_amount/a.day_diff) pay_amount, sum(a.discount_amount/a.day_diff) discount_amount FROM
(select meal_info_id, DATEDIFF(date(NOW()),date(pay_time)) day_diff, pay_amount/100 pay_amount, discount_amount/100 discount_amount from meal_subscribe
where `status`=4
and contact_name not like '%%测试%%') a
GROUP BY a.meal_info_id""", engine_subordinate)    

basket_no = pd.read_sql_query("""select meal_info_id, sum(1/DATEDIFF(date(NOW()),date(apply_time))) basket_cnt from meal_subscribe
where `status`!=4
and contact_name not like '%%测试%%'
GROUP BY meal_info_id""", engine_subordinate)

engine_cobub = create_engine('mysql+mysqldb://dba_read:oWXI8Hq2LHfKnS4t@localhost:3310/appmondb', connect_args={'charset':'utf8'})
phonecall_no = pd.read_sql_query("""select a.meal_info_id, sum(a.affinity_count) phonecall_cnt from
(select substring_index(label,',',1) meal_cate, substring_index(label, ',', -1) meal_info_id, 1/DATEDIFF(date(NOW()),date(clientdate)) affinity_count from appmon_eventdata
where event_id in (140, 141)
and insertdate < '2016-09-09'
and label is not null
and label !='')a
where a.meal_cate =2
GROUP BY a.meal_info_id""", engine_cobub)

favourite_no = pd.read_sql_query("""select object_id meal_info_id, sum(1/DATEDIFF(date(NOW()),date(create_time))) favourite_cnt from user_info_trajectory
where data_type =1
and object_type =20
and create_time < '2016-09-09'
GROUP BY object_id""", engine_subordinate)

forward_no = pd.read_sql_query("""select a.meal_info_id, sum(a.affinity_count) forward_cnt from
(select substring_index(label,',',1) meal_cate, substring_index(label, ',', -1) meal_info_id, 1/DATEDIFF(date(NOW()),date(clientdate)) affinity_count from appmon_eventdata 
where event_id in (150, 151)
and insertdate < '2016-09-09'
and label is not null
and label !='')a
where a.meal_cate =2
and a.meal_info_id is not null
and a.meal_info_id !=''
GROUP BY a.meal_info_id""", engine_cobub)

stepone_tb = pd.merge(pd.merge(pd.merge(pd.merge(purchase_no, basket_no, 'outer',  'meal_info_id'),
                      phonecall_no, 'outer', 'meal_info_id'),
                      favourite_no, 'outer', 'meal_info_id'),
                      forward_no, 'outer', 'meal_info_id')
stepone_tb = stepone_tb.fillna("ffill")
stepone_tb.describe()

from sklearn import linear_model
clf = linear_model.LassoLars(alpha=.1, fit_intercept = False)
clf.fit(X_train.to_sparse(), y_train) 











X_train = stepone_tb[['pay_amount', 'discount_amount', 'basket_cnt', 'phonecall_cnt', 'favourite_cnt', 'forward_cnt']]
#X_test = stepone_tb[['pay_amount', 'discount_amount', 'basket_cnt', 'phonecall_cnt', 'favourite_cnt', 'forward_cnt']][11001:16597]
y_train = stepone_tb['purchase_cnt']
#y_test = stepone_tb['purchase_cnt'][11001:16597]

from sklearn import linear_model
clf = linear_model.Lasso(alpha = 0.1)
clf.fit(X_train, X_test).predict(y_train) 
        
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X_train)
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier()
clf = clf.fit(X_train, y_train)

# Lasso
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
alpha = 0.1
clf = LassoCV(cv=20)
y_pred_lasso = clf.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

# ElasticNet
from sklearn.linear_model import ElasticNet
clf = ElasticNet(alpha=alpha, l1_ratio=0.7)
y_pred_enet = clf.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

plt.plot(enet.coef_, label='Elastic net coefficients')
plt.plot(lasso.coef_, label='Lasso coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()

        
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X_train, y_train)        



from time import time
from scipy import sparse
from scipy import linalg

from sklearn.datasets.samples_generator import make_regression
from sklearn.linear_model import Lasso

X, y = make_regression(n_samples=200, n_features=5000, random_state=0)
X_sp = sparse.coo_matrix(X)

Xs = X.copy()
Xs[Xs < 2.5] = 0.0
Xs = sparse.coo_matrix(Xs)
Xs = Xs.tocsc()

print("Matrix density : %s %%" % (Xs.nnz / float(X.size) * 100))

alpha = 0.1
sparse_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
dense_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)

t0 = time()
sparse_lasso.fit(Xs, y)
print("Sparse Lasso done in %fs" % (time() - t0))

t0 = time()
dense_lasso.fit(Xs.toarray(), y)
print("Dense Lasso done in %fs" % (time() - t0))

print("Distance between coefficients : %s"
      % linalg.norm(sparse_lasso.coef_ - dense_lasso.coef_))


#______________________________________________________________________________
# STEPTWO
import pandas as pd
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
from sqlalchemy import create_engine
engine_subordinate = create_engine('mysql+mysqldb://dbaread:8TvWzIH0n&yiqg3g@localhost:3308/yhouse', connect_args={'charset':'utf8'})

meal_info = pd.read_sql_query("""select a.meal_info_id, a.avg_person, a.price, a.interested_num, a.waiting_hour, b.cuisine_style from
(select id meal_info_id, (persons_max+persons_min)/2 avg_person, price/100 price, meal_host_id, interested_num, leading_minutes/60 waiting_hour from meal_info
where first_online_time < '2016-09-12'
and description not like '%%测试%%')a
LEFT JOIN
(select id host_info_id, cuisine_style from host_info)b
on a.meal_host_id = b.host_info_id""", engine_subordinate)

X_train = meal_info[['avg_person', 'price', 'interested_num', 'waiting_hour', 'cuisine_style']]
X_train.cuisine_style.fillna('无', inplace = True)
X_train_new = X_train.values
from kmodes import kprototypes
kproto = kprototypes.KPrototypes(n_clusters=20, init='Huang', verbose=1, 
                                 max_iter = 8, gamma = 0.1, n_init= 10)
clusters = kproto.fit_predict(X_train_new, categorical=[4])

print(kproto.cluster_centroids_)
print(kproto.enc_map_)
# Print training statistics
print(kproto.cost_)
print(kproto.n_iter_)

cluster = []
meal_info_id = [] 
for m, c in zip(meal_info['meal_info_id'].values, clusters):
    meal_info_id.append(m)
    cluster.append(c)
    print("meal_info_id: {}, cluster:{}".format(m, c))
steptwo = pd.DataFrame({'meal_info_id' : meal_info_id, 'cluster_no':cluster})    
    
cluster = []
cost_func = []   
for k in range(10, 31):
    kproto = kprototypes.KPrototypes(n_clusters=k, init='Huang', verbose=1, 
                                 max_iter = 8, gamma = 0.1, n_init= 5)
    clusters = kproto.fit_predict(X_train_new, categorical=[4])
    cost_func.append(kproto.cost_)

#[300363006.34467202,
# 297794383.8256194,
# 259874343.26544085,
# 249121839.61682907,
# 209654962.4331823,
# 234665245.2857531,
# 200304255.3046768,
# 183925687.86631921,
# 193861057.41716242,
# 205813789.50286216,
# 161128616.23209241,
# 163541100.44794267,
# 173036951.69915196,
# 150184458.15891957,
# 158939164.27866676,
# 147008197.89009956,
# 145468342.73912185,
# 140041598.97678849,
# 136449968.49525851]
# the optimal no of clustering is 20
    
    
    
    
    
    
    
    
    
    
    
    
syms = np.genfromtxt('D:\\Software\\kmodes-main\\kmodes-main\\examples/stocks.csv', dtype=str, delimiter=',')[:, 0]
X = np.genfromtxt('D:\\Software\\kmodes-main\\kmodes-main\\examples/stocks.csv', dtype=object, delimiter=',')[:, 1:]
X[:, 0] = X[:, 0].astype(float)    
    
kproto = kprototypes.KPrototypes(n_clusters=4, init='Cao', verbose=2)
clusters = kproto.fit_predict(X, categorical=[1, 2])    
    
print(kproto.cluster_centroids_)
print(kproto.enc_map_)
# Print training statistics
print(kproto.cost_)
print(kproto.n_iter_)

for s, c in zip(syms, clusters):
    print("Symbol: {}, cluster:{}".format(s, c))    
    
#______________________________________________________________________________
# STEPTHREE

import pandas as pd
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
from sqlalchemy import create_engine
engine_subordinate = create_engine('mysql+mysqldb://dbaread:8TvWzIH0n&yiqg3g@localhost:3308/yhouse', connect_args={'charset':'utf8'})
engine_cobub = create_engine('mysql+mysqldb://dba_read:oWXI8Hq2LHfKnS4t@localhost:3310/appmondb', connect_args={'charset':'utf8'})    
    
favourite_aff = pd.read_sql_query(""" select user_info_id, object_id meal_info_id, sum(1/DATEDIFF(date('2016-09-12'),date(create_time))) favourite_affinity from user_info_trajectory
where data_type =1 
and object_type = 20
and create_time < '2016-09-12'
GROUP BY user_info_id, meal_info_id""", engine_subordinate)

checkout_aff = pd.read_sql_query("""select user_info_id, meal_info_id, sum(1/DATEDIFF(date('2016-09-12'),date(pay_time))) check_out_affinity from meal_subscribe
where `status` = 4
and pay_time < '2016-09-12'
GROUP BY user_info_id, meal_info_id""", engine_subordinate)

basket_aff = pd.read_sql_query("""select user_info_id, meal_info_id, sum(1/DATEDIFF(date('2016-09-12'),date(apply_time))) basket_affinity from meal_subscribe
where `status` != 4
and apply_time < '2016-09-12'
GROUP BY user_info_id, meal_info_id""", engine_subordinate)

phonecall_aff = pd.read_sql_query("""select a.deviceid, a.meal_info_id, sum(a.affinity_count) phonecall_cnt from
(select deviceid, substring_index(label,',',1) meal_cate, substring_index(label, ',', -1) meal_info_id, 1/DATEDIFF(date('2016-09-12'),date(clientdate)) affinity_count from appmon_eventdata
where event_id in (140, 141)
and insertdate < '2016-09-12'
and label is not null
and label !='')a
where a.meal_cate =2
GROUP BY a.meal_info_id, a.deviceid""", engine_cobub)

forward_aff = pd.read_sql_query("""select a.deviceid, a.meal_info_id, sum(a.affinity_count) forward_cnt from
(select deviceid, substring_index(label,',',1) meal_cate, substring_index(label, ',', -1) meal_info_id, 1/DATEDIFF(date('2016-09-12'),date(clientdate)) affinity_count from appmon_eventdata 
where event_id in (150, 151)
and insertdate < '2016-09-12'
and label is not null
and label !='')a
where a.meal_cate =2
and a.meal_info_id is not null
and a.meal_info_id !=''
and a.meal_info_id != '(null)'
GROUP BY a.meal_info_id, a.deviceid""", engine_cobub)

click1_aff = pd.read_sql_query("""select a.deviceid, a.meal_info_id, sum(a.affinity_count) click_cnt1 from
(select deviceid, substring_index(label,',',1) meal_info_id, 1/DATEDIFF(date('2016-09-12'),date(clientdate)) affinity_count, substring_index(label,',',-2) meal_cate from appmon_eventdata
where event_id in (261, 321)
and length(label)-length(replace(label, ',', ''))=4
and insertdate < '2016-09-12'
and label is not NULL
and label !='')a
where a.meal_cate =2
and a.meal_info_id is not null
and a.meal_info_id !=''
and a.meal_info_id != '(null)'
GROUP BY a.meal_info_id, a.deviceid""", engine_cobub)


click2_aff = pd.read_sql_query("""select a.deviceid, a.meal_info_id, sum(a.affinity_count) click_cnt2 from
(select deviceid, substring_index(label,',',1) meal_info_id, 1/DATEDIFF(date('2016-09-12'),date(clientdate)) affinity_count, substring_index(label,',',-1) meal_cate from appmon_eventdata
where event_id in (261, 321)
and length(label)-length(replace(label, ',', ''))=3
and insertdate < '2016-09-12'
and label is not NULL
and label !='')a
where a.meal_cate =2
and a.meal_info_id is not null
and a.meal_info_id !=''
and a.meal_info_id != '(null)'
GROUP BY a.meal_info_id, a.deviceid""", engine_cobub)

deviced_userid = pd.read_sql_query("""select user_info_id, deviceid from user_info_base""",
                                   engine_subordinate)

stepthree = pd.merge(pd.merge(favourite_aff, checkout_aff, on = ['user_info_id', 'meal_info_id'],
                     how = 'outer'), basket_aff, 
                     on = ['user_info_id', 'meal_info_id'], how = 'outer')
                     
stepthree1 =pd.merge(pd.merge(pd.merge(phonecall_aff, forward_aff,
                                       on = ['deviceid', 'meal_info_id'], how = 'outer'),
                                       click1_aff, on = ['deviceid', 'meal_info_id'], 
                                       how = 'outer'), click2_aff, on = ['deviceid', 'meal_info_id'],
                                       how = 'outer')

stepthree = pd.merge(stepthree, deviced_userid, on = 'user_info_id', how = 'left')
stepthree = stepthree.drop('user_info_id', 1)
stepthree = pd.merge(stepthree1, stepthree, on = ['deviceid', 'meal_info_id'], how = 'outer')

stepthree.fillna(0, inplace = True)

stepthree['click_cnt'] = stepthree['click_cnt1'] + stepthree['click_cnt2']

stepthree['score'] = stepthree['phonecall_cnt'] * 0.5 +stepthree['forward_cnt']*0.5 +stepthree['click_cnt']*0.1 +stepthree['favourite_affinity'] * 0.5 +stepthree['click_cnt1']+stepthree['check_out_affinity']+stepthree['basket_affinity']* 0.3

stepthree_new = stepthree[['deviceid', 'meal_info_id', 'score']]





























   
    
