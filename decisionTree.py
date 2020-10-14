# 数据分析和准备
import pandas as pd
import numpy as np
import random as rnd

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 机器学习
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# 1、采集数据
print("## Acquire data")
# 我们首先将训练和测试数据集获取到Pandas DataFrames中。
# 我们还将这些数据集组合在一起，以对两个数据集运行某些操作。
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]


# 2、准备和清理的数据
print("## Prepare and clean data")
# 查看数据有哪些特征
print(train_df.columns.values.tolist())

# 预览数据
train_df.head()

# 确定特征的类型
# 2种基本特征类型： 1. 分类特征 2.连续要素
# 标识缺失值
print("########### train——data ###########")
train_df.info()
print('#'*40)
print("########### test——data ###########")
test_df.info()

# 在训练和测试集中，“机舱”功能的缺失值均超过70％。
# 这会对我们的预测模型产生负面影响。
# 因此，我们可以从数据集中消除此功能。
for df in combine:
    df.drop('Cabin', axis=1, inplace=True)

# 3、数据分析
print("## Data analysis")
# 训练集的数据摘要
train_df.describe()
# 根据这些信息，我们可以概述有关数据集的以下几点。
# 生存是具有0或1个值的分类特征。
# 约有38％的样本幸存
# 大多数乘客（> 75％）没有和父母或孩子一起旅行。

train_df.describe(percentiles=[.1, .2, .6, .7, .8, .9, .95, .99])
# 近30％的乘客有兄弟姐妹和/或配偶。
# 票价差异很大，只有极少的乘客（<1％）支付的费用高达512美元。
# 65-80岁年龄段的老年乘客很少（<2％）。



# 仅分类的特征
train_df.describe(include=['O'])
# 名称在数据集中是唯一的。（数量=唯一= 891）
# 性别变量具有两个可能的值，其中男性占65％。（顶部=男性，频率= 577 /数量= 891）
# 出发需要三个可能的值。大多数乘客使用的S端口
# 票证功能具有很高的重复值比率（22％）。

# 了解每个特征与“生存”的关联程度
# 在分类任务中，如果特征与目标具有良好的相关性，则该特征很有可能成为良好的特征。
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# 数据可视化
print("### Visualizing data")
sns.lineplot(x='SibSp', y='Survived', data=train_df)

grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# 4、再次清理
print("## Back to cleaning")
# 根据我们的简单数据分析，我们可以肯定地说年龄和登机是有用的特征。
# 因此，我们不能删除这些特征。
# 此外，票证特征具有很高的重复率，并且可能与生存率没有关联。
# 因此，它会对训练准确性产生负面影响。乘客编号也应删除。
# 我们可以用模式（mode）替换“ Embarked”特征的缺失值。
for df in combine:
    df.drop(columns=['Ticket', 'PassengerId'], inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode(dropna=True)[0], inplace=True)

train_df.info()

# 5、数据准备
print("## Data Preparation")
# 数据准备是机器学习过程中的关键步骤之一。
# 一些算法不能完全理解某些数据，或者与某些类型的数据不兼容。
# 例如，我们使用的决策树分类器不适用于字符串标签。
# 因此，我们必须使用编码方法将分类信息转换为数字信息。
#
# 现在让我们使用非常基本的编码方法，即数字编码。
# 数值编码非常简单：为每个类别分配一个任意数字。
# 我们可以为此使用替换方法。首先，我们必须创建一个地图。
num_encode = {
    'Sex': {'male': 0, "female": 1},
    'Embarked': {'S': 0, 'Q': 1, 'C': 2 }
}

# 然后，我们可以在pandas数据框中使用replace方法将标签替换为数字标签。
for df in combine:
    df.replace(num_encode, inplace=True)

# 在对数据集建模之前，我们必须将训练集划分为训练集和验证集。
# 这将有助于我们在训练过程中更准确地评估数据。
# 在确定了最终模型之后，我们使用测试数据集来衡量模型的性能。
# 这种方法将帮助我们减少过度拟合。
#
# 下周您可以了解有关评估方法的更多信息。
#
# 首先，我们必须将所有特征移至一个数据框，将目标移至一系列，但要保留索引。
# 因为这是实验的第一部分，所以我们仅采用分类特征。
# 在这里，我们可以将SibSp和Parch视为分类。
X = train_df[['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']]
y = train_df['Survived']

# 现在，X变量是属性集，y变量包含相应的标签。
#
# Scikit-Learn的model_selection库包含train_test_split方法。
# 我们可以使用它来将数据随机分为训练和验证集。
#
# test_size参数指定测试集的比率（此处为验证集）。
#
# 我们将30％的数据分成验证集，将70％的数据分成训练集。
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# 6、模型和预测
print("## Model and Predict")
# 现在，我们可以根据该数据训练决策树算法并做出预测。
# Scikit-Learn包含树库，该树库包含用于各种决策树算法的内置类/方法。
#
# 由于我们将在此处执行分类任务，因此在此示例中将使用DecisionTreeClassifier类。
tree = DecisionTreeClassifier(criterion='entropy')
# 正如您在课堂上所学到的，我们将使用entropy作为划分标准。
# 您还可以更改和试验许多其他参数。请在这里找到它们。
#
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#
# 此类的fit方法被称为在训练数据上训练算法。
# 我们的训练数据将作为参数传递给fit方法。

tree.fit(x_train, y_train)
# 现在我们可以对测试数据进行预测。为了进行预测，使用了DecisionTreeClassifier类的预测方法。

y_pred = tree.predict(x_test)
# 我们现在可以检查模型的准确性。模型准确度表示分类器正确的频率。

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))



