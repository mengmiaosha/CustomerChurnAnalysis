import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import os

# 设置输出路径
output_path = "D:\\CHEN 作品集\\德国宠物市场\\zooplus\\"
os.makedirs(output_path, exist_ok=True)  # 确保目录存在

# 设置随机种子保证结果可复现
np.random.seed(42)

# ======================
# 1. 生成模拟数据
# ======================
def generate_simulated_data(n_users=5000):
    """生成符合zooplus业务特性的模拟数据集"""
    # 基础用户信息
    user_ids = [f'USER_{i}' for i in range(1, n_users+1)]
    signup_dates = [datetime(2023,1,1) + timedelta(days=np.random.randint(0, 365)) for _ in range(n_users)]
    is_plus_member = np.random.choice([0, 1], size=n_users, p=[0.6, 0.4])
    
    # 模拟用户行为数据
    data = []
    for uid, signup, is_plus in zip(user_ids, signup_dates, is_plus_member):
        # 基础特征
        pet_type = np.random.choice(['dog', 'cat'], p=[0.6, 0.4])
        avg_order_value = np.abs(np.random.normal(loc=35 if is_plus else 25, scale=10))  # 确保正值
        
        # 续费相关日期
        renewal_date = signup + timedelta(days=365)
        
        # 生成续费前90天的订单行为
        orders_last_90d = np.random.poisson(lam=3.5 if is_plus else 1.8)
        avg_days_between_orders = np.abs(np.random.normal(loc=25 if is_plus else 45, scale=10))  # 确保正值
        
        # 生成关键行为指标
        last_order_days_ago = np.random.randint(10, 100)
        browsing_freq = np.abs(np.random.normal(loc=2.5 if is_plus else 0.8, scale=1.2))  # 确保正值
        
        # 生成标签 - 是否流失
        churn = 0
        if is_plus:
            # 付费会员流失逻辑：最后订单>60天前 且 近90天订单<2
            if last_order_days_ago > 60 and orders_last_90d < 2:
                churn = 1 if np.random.random() > 0.3 else 0  # 70%概率流失
            else:
                churn = 1 if np.random.random() > 0.8 else 0  # 20%概率流失
        else:
            # 免费用户流失逻辑：最后订单>90天前
            churn = 1 if last_order_days_ago > 90 else 0
            
        data.append([
            uid, signup, is_plus, pet_type, avg_order_value,
            orders_last_90d, last_order_days_ago, avg_days_between_orders,
            browsing_freq, renewal_date, churn
        ])
    
    # 创建DataFrame
    columns = [
        'user_id', 'signup_date', 'is_plus_member', 'pet_type',
        'avg_order_value', 'orders_last_90d', 'last_order_days_ago',
        'avg_days_between_orders', 'browsing_freq', 'renewal_date', 'churn'
    ]
    return pd.DataFrame(data, columns=columns)

# 生成模拟数据
df = generate_simulated_data()
print("模拟数据示例:")
print(df.head())

# 保存模拟数据到Excel
excel_path = os.path.join(output_path, "zooplus_simulated_data.xlsx")
df.to_excel(excel_path, index=False)
print(f"\n模拟数据已保存至: {excel_path}")

# ======================
# 2. 沉睡客户分析
# ======================
def analyze_dormant_users(df, output_path):
    """识别和分析沉睡客户特征"""
    # 定义沉睡客户：最后订单>60天前的付费会员
    df['is_dormant'] = ((df['last_order_days_ago'] > 60) & 
                        (df['is_plus_member'] == 1)).astype(int)
    
    dormant_percent = df['is_dormant'].mean()
    print(f"\n沉睡客户占比: {dormant_percent:.1%}")
    
    # 分析沉睡客户特征
    dormant_df = df[df['is_dormant'] == 1]
    print("\n沉睡客户特征分析:")
    print(dormant_df[['orders_last_90d', 'last_order_days_ago', 
                     'avg_days_between_orders', 'browsing_freq']].describe())
    
    # 可视化关键指标 - 分别保存4个独立图表
    
    # 1. 近90天订单量分布
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dormant_df, x='orders_last_90d', bins=10, kde=True)
    plt.title('近90天订单量分布')
    plt.axvline(x=1.5, color='r', linestyle='--', label='干预阈值')
    plt.legend()
    order_dist_path = os.path.join(output_path, "dormant_order_distribution.png")
    plt.savefig(order_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"订单量分布图已保存至: {order_dist_path}")
    
    # 2. 客单价对比（沉睡vs活跃）
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='is_dormant', y='avg_order_value')
    plt.title('客单价对比')
    plt.xticks([0, 1], ['活跃用户', '沉睡用户'])
    spend_box_path = os.path.join(output_path, "dormant_spend_boxplot.png")
    plt.savefig(spend_box_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"客单价箱线图已保存至: {spend_box_path}")
    
    # 3. 多维度行为特征分析
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=dormant_df, x='last_order_days_ago', y='browsing_freq', 
                   hue='pet_type', size='avg_order_value', alpha=0.7)
    plt.title('沉睡客户行为特征')
    plt.xlabel('距最后订单天数')
    plt.ylabel('浏览频率')
    behavior_path = os.path.join(output_path, "dormant_behavior_scatter.png")
    plt.savefig(behavior_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"行为特征散点图已保存至: {behavior_path}")
    
    # 4. 宠物类型与沉睡比例
    plt.figure(figsize=(10, 6))
    cross_tab = pd.crosstab(df['pet_type'], df['is_dormant'], normalize='index')
    cross_tab.plot(kind='bar', stacked=True)
    plt.title('宠物类型与沉睡比例')
    plt.ylabel('比例')
    plt.legend(['非沉睡', '沉睡'], title='状态')
    pet_type_path = os.path.join(output_path, "dormant_pet_type_stacked.png")
    plt.savefig(pet_type_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"宠物类型堆叠图已保存至: {pet_type_path}")
    
    # 高级分析：沉睡客户与流失的关系
    plt.figure(figsize=(10, 6))
    heatmap_data = pd.crosstab(df['is_dormant'], df['churn'], normalize='index')
    sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='Blues')
    plt.title('沉睡状态与流失率关系')
    plt.xlabel('是否流失')
    plt.ylabel('是否沉睡')
    dormant_churn_path = os.path.join(output_path, "dormant_churn_heatmap.png")
    plt.savefig(dormant_churn_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"沉睡流失关系热力图已保存至: {dormant_churn_path}")
    
    return df

df = analyze_dormant_users(df, output_path)

# ======================
# 3. 流失预测模型
# ======================
def build_churn_model(df, output_path):
    """构建并评估客户流失预测模型"""
    # 数据预处理
    df = df.copy()
    df = pd.get_dummies(df, columns=['pet_type'], drop_first=True)
    
    # 特征工程
    features = df.drop(['user_id', 'signup_date', 'renewal_date', 'is_dormant', 'churn'], axis=1)
    target = df['churn']
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )
    
    # 构建模型管道
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # 参数网格
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5]
    }
    
    # 网格搜索
    print("\n开始网格搜索...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("网格搜索完成!")
    
    # 最佳模型评估
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    
    print("\n最佳模型参数:", grid_search.best_params_)
    print(f"测试集AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # 保存分类报告
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(output_path, "classification_report.xlsx")
    report_df.to_excel(report_path)
    print(f"分类报告已保存至: {report_path}")
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend()
    roc_path = os.path.join(output_path, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"ROC曲线已保存至: {roc_path}")
    
    # 特征重要性
    rf_model = best_model.named_steps['classifier']
    importances = pd.Series(rf_model.feature_importances_, index=features.columns)
    
    plt.figure(figsize=(10, 6))
    importances.sort_values(ascending=True).plot(kind='barh')  # 横向条形图更清晰
    plt.title('特征重要性')
    plt.xlabel('重要性得分')
    plt.tight_layout(pad=2.0)
    feature_imp_path = os.path.join(output_path, "feature_importance.png")
    plt.savefig(feature_imp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征重要性图表已保存至: {feature_imp_path}")
    
    # 保存特征重要性数据
    importance_df = importances.sort_values(ascending=False).reset_index()
    importance_df.columns = ['特征', '重要性']
    importance_path = os.path.join(output_path, "feature_importance.xlsx")
    importance_df.to_excel(importance_path, index=False)
    print(f"特征重要性数据已保存至: {importance_path}")
    
    # 保存预测结果
    test_results = X_test.copy()
    test_results['实际流失'] = y_test
    test_results['预测流失概率'] = y_pred_proba
    test_results['预测流失'] = y_pred
    results_path = os.path.join(output_path, "prediction_results.xlsx")
    test_results.to_excel(results_path, index=False)
    print(f"预测结果已保存至: {results_path}")
    
    return best_model, features.columns.tolist()

# 构建预测模型
model, feature_names = build_churn_model(df, output_path)

print("\n分析完成！所有数据和图表已保存至指定路径")
print(f"输出路径: {output_path}")
