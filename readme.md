将代码文件和训练及测试数据放到同一个文件夹中，执行python main.py即可生成提交的结果文件
最后的分数是0.82296, 排名379
build_model.py 用来调节参数构造模型
gen_features.py用来提取数据特征
但main.py 中并未调用gen_features.py中全部函数，有些特征有可能导致过拟合，因此未拼接到训练数据中 
