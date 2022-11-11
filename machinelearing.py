import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold

#读取数据
traindata = pd.read_csv(r'H:\train.csv')#读取训练集
testdata = pd.read_csv(r'H:\evaluation_public.csv')#读取测试集
df = pd.concat([traindata, testdata])#合并训练集与测试集

#对op_datetime进行特征分割，提取时间特征
df['op_datetime'] = pd.to_datetime(df['op_datetime'])#将op_datetime转化为datetime类型
df['hour'] = df['op_datetime'].dt.hour#根据op_datetime提取小时
df['weekday'] = df['op_datetime'].dt.weekday#根据op_datetime提取星期

df = df.sort_values(by=['user_name', 'op_datetime']).reset_index(drop=True)#按用户名、op_datetime排序
df['ts'] = df['op_datetime'].values.astype(np.int64) // 10 ** 9#将datetime类型转化为numpy的整形
df['ts1'] = df.groupby('user_name')['ts'].shift(1)#根据名字分组，并进行错位
df['ts2'] = df.groupby('user_name')['ts'].shift(2)#根据名字分组，并进行错位
df['ts_diff1'] = df['ts1'] - df['ts']#计算其和上一次访问时的间隔
df['ts_diff2'] = df['ts2'] - df['ts']#计算其和上二次访问时的间隔

df['hour_sin'] = np.sin(df['hour']/24*2*np.pi)#把小时放到正弦上并对其进行正余弦化
df['hour_cos'] = np.cos(df['hour']/24*2*np.pi)#把小时放到余弦上

LABEL = 'is_risk'
#类型特征
cat_f = ['user_name', 'department', 'ip_transform', 'device_num_transform', 'browser_version', 'browser',
          'os_type', 'os_version', 'ip_type', 'op_city', 'log_system_transform', 'url',]

#对特征类型进行编码，转换为离散的数
for f in cat_f:
    le = LabelEncoder()
    df[f] = le.fit_transform(df[f])
    df[f+'_ts_diff_mean'] = df.groupby([f])['ts_diff1'].transform('mean')
    df[f+'_ts_diff_std'] = df.groupby([f])['ts_diff1'].transform('std')

#区分训练集和测试集
traindata = df[df[LABEL].notna()].reset_index(drop=True)
testdata = df[df[LABEL].isna()].reset_index(drop=True)

#模型训练
feats = [f for f in testdata if f not in [LABEL, 'id','op_datetime', 'op_month', 'ts', 'ts1', 'ts2']]
print(feats)
print(traindata[feats].shape, testdata[feats].shape)
params = {
    'learning_rate': 0.05,#学习速率
    'boosting_type': 'gbdt',#设置提升类型
    'objective': 'binary',#目标函数
    'metric': 'auc',#评估函数
    'num_leaves': 64,#叶子节点数
    'verbose': -1,#显示信息
    'seed': 2222,#随机种子数
    'n_jobs': -1,

    'feature_fraction': 0.8,#建树特征选择比例
    'bagging_fraction': 0.9,#建树样本采样比例
    'bagging_freq': 4,# 意味着每 k 次迭代执行bagging
}

seeds = [2222]
oof = np.zeros(len(traindata))
importance = 0
pred_y = pd.DataFrame()
score = []
for seed in seeds:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)#k折交叉验证
    for fold, (train_idx, val_idx) in enumerate(kf.split(traindata[feats], traindata[LABEL])):
        print('-----------', fold)
        #创建成lgb特征的数据集格式
        train = lgb.Dataset(traindata.loc[train_idx, feats],
                            traindata.loc[train_idx, LABEL])
        val = lgb.Dataset(traindata.loc[val_idx, feats],
                          traindata.loc[val_idx, LABEL])
        #训练 cv and train
        model = lgb.train(params, train, valid_sets=[val], num_boost_round=20000,
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(2000)])
        #进行预测
        oof[val_idx] += model.predict(traindata.loc[val_idx, feats]) / len(seeds)
        pred_y['fold_%d_seed_%d' % (fold, seed)] = model.predict(testdata[feats])
        #获取特征重要性
        importance += model.feature_importance(importance_type='gain') / 5
        #计算auc的值即模型评价指标
        score.append(auc(traindata.loc[val_idx, LABEL], model.predict(traindata.loc[val_idx, feats])))

#对特征按重要性进行排序
feats_importance = pd.DataFrame()
feats_importance['name'] = feats
feats_importance['importance'] = importance
print(feats_importance.sort_values('importance', ascending=False)[:30])

#输出模型评价指标
traindata['oof'] = oof
print(np.mean(score), np.std(score))
#写入测试集结果
score = np.mean(score)
testdata[LABEL] = pred_y.mean(axis=1).values
testdata = testdata.sort_values('id').reset_index(drop=True)
#读取提交格式，提交结果
sub = pd.read_csv(r'H:\submit_sample.csv')
sub['ret'] = testdata[LABEL].values
sub.columns = ['id', LABEL]
sub.to_csv(time.strftime(r'C:\Users\25144\Desktop\ans1.csv')+'%.5f.csv'%score, index=False)
