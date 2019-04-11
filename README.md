# sentence_similarity

句子相似度计算包括以下几种方法：
- 编辑距离计算
- 杰卡德系数计算
- TF 计算
- TFIDF 计算
- Word2Vec 计算


## 编辑距离计算
编辑距离，英文叫做 Edit Distance，又称 Levenshtein 距离，是指两个字串之间，由一个转成另一个所需的最少编辑操作次数，如果它们的距离越大，说明它们越是不同。许可的编辑操作包括将一个字符替换成另一个字符，插入一个字符，删除一个字符。

例如我们有两个字符串：string 和 setting，如果我们想要把 string 转化为 setting，需要这么两步：

- 第一步，在 s 和 t 之间加入字符 e。
- 第二步，把 r 替换成 t。
所以它们的编辑距离差就是 2，这就对应着二者要进行转化所要改变（添加、替换、删除）的最小步数。

## 杰卡德系数计算
杰卡德系数，英文叫做 Jaccard index, 又称为 Jaccard 相似系数，用于比较有限样本集之间的相似性与差异性。Jaccard 系数值越大，样本相似度越高。

实际上它的计算方式非常简单，就是两个样本的交集除以并集得到的数值，当两个样本完全一致时，结果为 1，当两个样本完全不同时，结果为 0。

## TF 计算
第三种方案就是直接计算 TF 矩阵中两个向量的相似度了，实际上就是求解两个向量夹角的余弦值，就是点乘积除以二者的模长，公式如下：
```
cosθ=a·b/|a|*|b|
```
上面我们已经获得了 TF 矩阵，下面我们只需要求解两个向量夹角的余弦值就好了

## TFIDF 系数
另外除了计算 TF 系数我们还可以计算 TFIDF 系数，TFIDF 实际上就是在词频 TF 的基础上再加入 IDF 的信息，IDF 称为逆文档频率。

http://www.ruanyifeng.com/blog/2013/03/tf-idf.html

## Word2Vec 计算
Word2Vec，顾名思义，其实就是将每一个词转换为向量的过程。

这里采用训练好的模型包括：
- [64维的Word2Vec模型](https://pan.baidu.com/s/1p8-_fP6N4C6lgjDLLe4CSg): 12g新闻数据，20g百度百科数据，90g小说数据 （提取码：hsqy）
- [300维的Word2Vec模型](https://pan.baidu.com/s/1p_kUVc0StXrUTijQy0PtKg): [搜狐新闻数据](http://www.sogou.com/labs/resource/cs.php)(提取码：frip)

在获取 Sentence Vector 的时候，我们首先对句子进行分词，然后对分好的每一个词获取其对应的 Vector，然后将所有 Vector 相加并求平均，这样就可得到 Sentence Vector 了，然后再计算其夹角余弦值即可。
