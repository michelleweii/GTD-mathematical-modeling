# GTD-mathematical-modeling
2015 "华为杯" Mathematical modeling 

## 任务1 依据危害性对恐怖袭击事件分级
对灾难性事件比如地震、交通事故、气象灾害等等进行分级是社会管理中的重要工作。通常的分级一般采用主观方法，由权威组织或部门选择若干个主要指标，强制规定分级标准，如我国《道路交通事故处理办法》第六条规定的交通事故等级划分标准，主要按照人员伤亡和经济损失程度划分。
但恐怖袭击事件的危害性不仅取决于人员伤亡和经济损失这两个方面，还与发生的时机、地域、针对的对象等等诸多因素有关，因而采用上述分级方法难以形成统一标准。请你们依据附件1以及其它有关信息，结合现代信息处理技术，借助数学建模方法建立基于数据分析的量化分级模型，将附件1给出的事件按危害程度从高到低分为一至五级，列出近二十年来危害程度最高的十大恐怖袭击事件，并给出表1中事件的分级。

表1 典型事件危害级别

事件编号 | 危害级别
--|--
200108110012|
200511180002|
200901170021|
201402110015|
201405010071| 
201411070002|	  
201412160041|	
201508010015|	  
201705080012| 	

首先数据预处理
恐怖袭击事件的数据中有些特征与问题求解无关且有数据不一致等诸多问题，严重影响到数据挖掘建模的执行效率和结果。数据预处理过程如下：
第一步，计算每个属性的空值总数和百分比。结果显示，许多属性的缺失值超过50％，对于缺失值较多的属性，由此选择缺失值小于20％且不与其他属性相似的属性列表用于接下来的建模。
第二步，因为对缺失值进行分类会给模型造成一定的分类障碍，所以我们修复了缺失值。其中包括原始数据空白值，-9，-99。为了保持一致性，规定-1用于数字的分类属性，Unknown用于文本的分类属性，数字属性将替换为NAN。部分属性由于平均值不稳定且受异常值的影响较大，我们采用中值进行插补。
第三步，由于很多属性是用Yes/No/Unknown进行标记的，这种标记方法不利于我们进行数据分析，用1，0，-1分别表示“Yes”，“No”和“Unknow“。 不止这些，用数字标签替换数据集中的其他文本属性以改进原始数据集。
第四步，由于不同评价指标往往具有不同的量纲，数值间的差别可能很大，不进行处理可能会影响到数据分析的结果。为了消除指标之间的量纲和取值范围差异的影响，需要进行标准化处理，将数据按照比例进行缩放，使其落入一个特定的区域，便于进行综合分析。实验预处理部分采用的是最小-最大规范化，对原始数据进行线性变换，将数值映射到[0,1]之间。进行数据规范化的特征有：'nperpcap', 'nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte'。
第五步：用随机森林模型挑选出来了前20个重要特征，根据互联网查找资料所显示的“恐怖袭击事件”的广泛定义，作为接下来每个任务的具体求解参照，删除了不属于此范畴内的特征，利用符合每个任务题意的特征进行了建模。前20个重要特征有：

随机森林产生的20个特征

--|--|--|--|--
iyear	| region_txt_South Asia	| INT_IDEO_txt_YES	| natlty1_txt_Iraq | INT_ANY_txt_YES
region_txt_Middle East&North Africa | country_txt_Iraq | country_txt_Somalia | iday|imonth
region_txt_Sub-Saharan Africa | INT_LOG_txt_UKNOWN | INT_IDEO_txt_UKNOWN | nkill | INT_ANY_txt_UKNOWN
country_txt_Pakistan | natlty1_txt_Somalia | natlty1_txt_India | natlty1_txt_Pakistan | natlty1_txt_Nigeria


## 任务2 依据事件特征发现恐怖袭击事件制造者
附件1中有多起恐怖袭击事件尚未确定作案者。如果将可能是同一个恐怖组织或个人在不同时间、不同地点多次作案的若干案件串联起来统一组织侦査，有助于提高破案效率，有利于尽早发现新生或者隐藏的恐怖分子。请你们针对在2015、2016年度发生的、尚未有组织或个人宣称负责的恐怖袭击事件，运用数学建模方法寻找上述可能性，即将可能是同一个恐怖组织或个人在不同时间、不同地点多次作案的若干案件归为一类，对应的未知作案组织或个人标记不同的代号，并按该组织或个人的危害性从大到小选出其中的前5个，记为1号-5号。再对表2列出的恐袭事件，按嫌疑程度对5个嫌疑人排序，并将结果填入下表（表中样例的意思是：对事件编号为XX的事件，3号的嫌疑最大，其次是4号，最后是5号），如果认为某嫌疑人关系不大，也可以保留空格。

表2 恐怖分子关于典型事件的嫌疑度

 1号嫌疑人 |	2号嫌疑人	| 3号嫌疑人	| 4号嫌疑人	| 5号嫌疑人
--|--|--|--|--
样例XX | 4	| 3	| 1 | 2 |	5
201701090031|					
201702210037|					
201703120023|					
201705050009|					
201705050010|					
201707010028|					
201707020006|					
201708110018|					
201711010006|					
201712010003|					

由于数据集中部分数据的缺失，有多起恐怖袭击事件尚未确定作案者。任务二要求确定2015年、2016年度发生的、尚未由组织或个人宣称负责的恐怖事件，按照一定规律找出相似事件的类别，将其归为一类。在此基础上，选出组织或个人的危害程度排名前5的完成表2任务。本任务首先对未知事件通过随机森林算法建模进行分类任务，通过数据分析得出排名前5的组织，标记为1-5号，之后通过k-近邻算法建模找到与未知恐怖事件有嫌疑的5个嫌疑人。

## 任务3 对未来反恐态势的分析
对未来反恐态势的分析评估有助于提高反恐斗争的针对性和效率。请你们依据附件1并结合因特网上的有关信息，建立适当的数学模型，研究近三年来恐怖袭击事件发生的主要原因、时空特性、蔓延特性、级别分布等规律，进而分析研判下一年全球或某些重点地区的反恐态势，用图/表给出你们的研究结果，提出你们对反恐斗争的见解和建议。

针对该问题，有多种方法可以达到时间序列预测的结果。我们采用了时间序列预测框架Prophet[2]。对由数据分析结果显示的重点地区进行了下一年的恐怖袭击态势的预测。针对恐怖袭击发生的规律从主要原因、蔓延特性、级别分布等方面做了数据分析可视化，从而实现了此任务的目标，并在结果分析中对反恐斗争提出了见解和建议。
（1） 时间序列模型
	在针对于该问题的解决，应用了Facebook开源的时间序列预测算法。Prophet是一种基于加性模型预测时间序列数据的程序，其中非线性趋势能够拟合每一年，每一周和每一日。它对于缺失数据和趋势的变化是非常有效的，通常很好地处理异常值，并且适用于具有强烈季节性影响的历史数据的时间序列。因为从数据分析中得出易发生恐怖袭击的国家大多数都是信奉伊斯兰教的国家，教徒们通常会根据节日做相应的祷告活动，我们大胆推断发生恐怖袭击事件的日期与他们节假日之间有着一定的关联，结果显示也证实了我们猜想的正确性。Prophet是一个可加回归模型，也就是把模型分为趋势模型、周期模型、节假日以及随机噪声的叠加。它由四个组成部分：
1.	一个分段的线性或逻辑增长曲线趋势。Prophet通过提取数据中的转变点，自动检测趋势变化。
2.	一个按年的周期组建，使得傅里叶级数建模而成。
3．一个按周的周期组建，使得虚拟变量建模而成。
4. 用户设置重要节日表。


## 任务4 数据的进一步利用
你们认为通过数学建模还可以发挥附件1数据的哪些作用？给出你们的模型和方法。


（1）	利用数据集中所显示的恐怖袭击中使用的武器，按照其类型进行分类，此问题提出的依据如图所示
图片显示爆炸袭击，枪炮，以及不知具体武器的攻击类型占了前三位。所以根据显示结果，利用附件1的数据建立了武器分类模型，以便预测出来武器攻击类型以做好未来针对该武器类型的防范。
（2） 自然语言处理是人工智能最早的研究领域之一，通过对附件1数据分析得知有summary和country_txt两个特征。利用Seq2Seq模型对这两个属性进行数学建模。我们所要解决的问题是根据summary内容预测出country_txt中的内容。在数据分析领域，数据集严重影响着分析结果，每一条数据都很珍贵，如果country_txt内容丢失，我们完全可以从summary中预测出结果，而不必删除缺失值的数据。


