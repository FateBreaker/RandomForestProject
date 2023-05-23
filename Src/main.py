import pandas as pd
import numpy as np

#导入数据并拆分为训练集和测试集
# filename = 'D:/churn.csv'
filename = 'Soure\churn.csv'
data = pd.read_csv(filename)
col_name = list(data.columns)#获取所有列名

x_col = col_name
col_drop=['State','Account Length','Area Code','Phone', 'Churn?']#一些无意义的列，以及标签列'Churn?'
for i in col_drop:
    x_col.remove(i)
#print(x_col)

#查看是否有缺失值
print(data.isnull().any())

#数据预处理
#查看变量的取值
data['Intl Plan'].value_counts()
data['VMail Plan'].value_counts()
data['Intl Plan']=data['Intl Plan'].map({'yes':1,'no':0})
data['VMail Plan']=data['VMail Plan'].map({'yes':1,'no':0})


#拆分数据集
from sklearn.model_selection import train_test_split
X = data[x_col]

y = data['Churn?']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=33)

#使用随机森林分类
from sklearn.ensemble import RandomForestClassifier
#不断增大基分类器数量，查看分类准确率变换过程，寻找最佳的基分类器数
from sklearn.model_selection import cross_val_score
score_lst = []
# 修改分类器数量
n = 20
for i in range(n):
    rfc = RandomForestClassifier(n_estimators=i+1)
    rfc_score = cross_val_score(rfc,X_train,y_train).mean()  #cv=10
    score_lst.append(rfc_score)
k = score_lst.index(max(score_lst)) + 1
print('最佳得分为{}，对应的基分类器数量为{}'.format(max(score_lst),k))
#运行结果：最佳得分为0.9583177518085698，对应的基分类器数量为98

#基分类器数量与对应的分类准确率变换过程可视化
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Simhei']#正常显示中文
plt.figure(figsize=[20,5])
plt.plot(range(1,n+1),score_lst)
plt.xlabel('基分类器数量k')
plt.ylabel('模型预测准确率')
plt.show()


#使用找到的最佳基分类器数构建最佳随机森林
rfc = RandomForestClassifier(n_estimators=k,random_state=2,oob_score=True)
#oob_score=True表示使用袋外数据进行模型效果评估
rfc = rfc.fit(X_train,y_train)
print(rfc.estimators_)#用列表存储生成的62个基分类器信息
'''
[DecisionTreeClassifier(max_features='auto', random_state=1872583848),
DecisionTreeClassifier(max_features='auto', random_state=794921487),
DecisionTreeClassifier(max_features='auto', random_state=111352301),
DecisionTreeClassifier(max_features='auto', random_state=1853453896),
DecisionTreeClassifier(max_features='auto', random_state=213298710),
DecisionTreeClassifier(max_features='auto', random_state=1922988331),
DecisionTreeClassifier(max_features='auto', random_state=1869695442),
DecisionTreeClassifier(max_features='auto', random_state=2081981515),
DecisionTreeClassifier(max_features='auto', random_state=1805465960),
DecisionTreeClassifier(max_features='auto', random_state=1376693511),
DecisionTreeClassifier(max_features='auto', random_state=1418777250),
DecisionTreeClassifier(max_features='auto', random_state=663257521), ……]#共62个
'''
#查看森林中基分类器的随机状态值
for i in range(len(rfc.estimators_)):
    print(i,rfc.estimators_[i].random_state)

print (rfc.oob_score_)#0.9509836612204068
'''
oob_score_:
Score of the training dataset obtained using an out-of-bag estimate.
即，使用袋外样本来估计模型的泛化能力。
随机森林采用有放回采样，大约36.8%的样本会没有被采样到，我们常常称之为袋外数据(Out Of Bag,OOB)，
这些数据既不属于训练集，也不属于测试集，因此可以用来检测模型的泛化能力。
'''

print ('col:',x_col) #打印特征项，方便下面查看特征项的重要性
print(list((rfc.feature_importances_).flatten()))#系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大
#将特征名和特征重要性保存为1个数据框
feature_importance_df = pd.DataFrame({'featurename':x_col,'importance':np.abs(rfc.feature_importances_)})
feature_importance_df = feature_importance_df.sort_values(by='importance',ascending=False)#根据特征重要性，进行降序排列
#print(feature_importance_df)

#只保留特征重要性大于0.0001的记录
feature_importance_df = feature_importance_df[feature_importance_df['importance']>0.0001]
print(feature_importance_df)

#将特征重要性绘制成柱状图
plt.figure(figsize=[5,5])
import matplotlib.pyplot as plt
location = np.arange(len(feature_importance_df['featurename']))#即np.arange(3),生成0、1、2，作为y轴的坐标值
#print(location)
importance = feature_importance_df['importance']

#barh用于绘制水平状的柱状图
plt.barh(y=location, width=importance)
'''
y=location设置柱状图的y坐标.
width=importances 表示柱状图的长度由特征重要性的值确定，特征越重要，柱状图越长。
'''
plt.yticks(ticks=location, labels=feature_importance_df['featurename'])#ticks用于指定y坐标轴位置，labels指定在坐标轴上显示的标签内容
plt.tick_params(labelsize=6)
plt.xlabel('Importances')
plt.xlim(0, 0.5)
plt.title('Features Importances')
plt.show()

#模型评估
score = cross_val_score(estimator=rfc, X=X_test,y=y_test,scoring='accuracy',cv=3)#进行3次交叉验证，得到3个精确度得分
print(score)
print(score.mean())
print ('测试集准确率',rfc.score(X_train,y_train))
print ('测试集准确率',rfc.score(X_test,y_test))
from sklearn.metrics import classification_report
y_predict = rfc.predict(X_test)
print (classification_report(y_predict,y_test))


#根据第一次随机森林得到的特征重要性，筛选重要特征构建第二个随机森林
from sklearn.feature_selection import SelectFromModel
threshold = min(feature_importance_df['importance'])+0.01#可以人为设置特征重要性的阈值，阈值越大，最终生成的新决策树被保留的特征越少
rfe = SelectFromModel(estimator=rfc,threshold=threshold)
'''
estimator:一个已经拟合好的基分类器，这里是第一次生成的随机森林，SelectFromModel将对该分类器进行完善，仅保留重要性大于threshold的特征
'''
rfe.fit(X_train, y_train)
X_train_new = rfe.transform(X_train)
X_test_new = rfe.transform(X_test)
'''
在选择好大于阈值的特征后，使用SelectFromModel类中的transform()方法，将X（特征集）进行变换，仅保留被选中的特征。
----Reduce X to the selected features.
转换后的特征集数据格式为ndarray，没有表头。
'''
#依然使用最开始得到的最佳基分类器数量，构建新的随机森林
rfc_new = RandomForestClassifier(n_estimators=k,random_state=2,oob_score=True)
rfc_new.fit(X_train_new, y_train)#重新拟合随机森林
#模型评价
print('选取SelectFromModel特征之后的训练结果：', rfc_new.score(X_train_new, y_train))
print('选取SelectFromModel特征之后的准确性：', rfc_new.score(X_test_new, y_test))
'''
可以发现，即使删除了几个特征，整体的分类准确率并没有降低，那些特征是真的该删！
'''
print (rfc.oob_score_)
score = cross_val_score(estimator=rfc, X=X_test_new,y=y_test,scoring='accuracy',cv=3)#进行3次交叉验证，得到3个精确度得分
print(score)
print(score.mean())
y_predict_new = rfc_new.predict(X_test_new)
print (classification_report(y_predict_new,y_test))