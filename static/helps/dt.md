 # 文本分类——决策树模型  
 ## 1 决策树模型的训练  
一般决策树的训练过程是先建立一棵大规模的树结构，然后再对这个树进行剪枝，知道到达合适的规模和分类效率。  
剪枝是决策树不可缺少的一步，否则在一棵大规模的树上进行分类判定，很容易就会出现过适应问题，特别是分类器基于训练集的一些弱属性上做决策时，经常出现过适应问题。  
### 1.1     分支准则  
决策树的分支准则是决定当前树节点选择何种属性作为当前训练数据的分支属性。一般的，分支准则采用信息增益原则。  
信息增益是衡量一个特征能给分类系统带来多少信息，带来的越多，那么这个属性就越重要。   
$$IG(T) = H(C) – H(C|T)  $$
其中 T为属性，C为分类类别，H为熵。 从公式可以看出因为H(C)大小时固定的，所以当信息增益越大，那么H(C|T)就越小。从这个角度来看，信息增益一个直观的的解释就是，对于属性T的引入使得整个系统不确定性减小。下面用搜狗实验室的语料举一个例子来介绍怎么计算信息增益。    
  从搜狗实验室中取财经文章1001篇为正例，娱乐类文章1208篇为负例，分词统计各个词的文档频次，摘抄几个如下    
--------------------------------------
Term | 正例文档出现的次数 |  负例文档出现的次数
---|---|--  
观众|5 |492
市场|429 |48
公司|281 |0
导演|2 |363
演员|1 |341
企业|281 |14

 
假设当前选择Term为 “市场” 那么信息增益会是多少呢？
  $$IG(市场) = H(商业) + H(娱乐) – H( 娱乐|市场) – H(商业|市场)$$  
      
代入公式即可以求的 “市场”带来的信息增益为0.174389，类似的可以算出其他Term的信息增益分别为  
$$IG(观众) = 0.215399 ，IG(公司) = 0.160949，IG(导演) = 0.1549991， IG(演员) = 0.147911， IG(企业) = 0.129144.$$  
从这个结果可以发现“观众”的增益是最大的，所以当前节点应该选择“观众”作为分支属性，包含“观众”的文档被划归到左子树，而其他的文档被划归到右子树。  
## 1.2     停止准则
停止准则也可以成为剪枝准则，一般停止准则判断条件为：决策树的当前节点的所有数据都具备相同的类别。  
在我的代码中采用了3个条件来停止迭代  
条件一： 当前节点的文档数 <= 10
条件二：文档都属于同一个类别
条件三：在计算信息增益时，发现左右子树任意一节点的文档数不足5。
## 1.3代码实现
此代码只是简单的实现了一个根据词是否存在的决策树，剪枝条件也没有经过严格的调整。还有比较大的优化空间。

## 1.4  模型评价
Positive recall :91.106439  
Positive precision :96.230291  
Accuary : 92.071295  
从这个数据来看，这个简单的模型效果还是不错，不过估计会比svm之类的效果略差。
## 1.5  模型的优缺点
优点：决策树最大的优点就是结构清晰，容易理解。在构建树的过程中如果出现问题很容易调试——直接将树打印出来问题就一目了然了。
缺点：模型容易出现过拟合问题，需要好的剪枝策略，才能达到好的分类效果。
                                               
                                              
 
 