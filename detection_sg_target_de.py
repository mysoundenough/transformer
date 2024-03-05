import pandas as pdimport torchimport torch.nn as nnfrom numpy import * import numpy as npimport matplotlib.pyplot as pltfrom matplotlib import font_managerfrom torch.utils.data import DataLoader,TensorDatasetfrom sklearn.preprocessing import StandardScaler # 标准化数据import timefrom sklearn.model_selection import train_test_splitfrom matplotlib import font_managerimport seaborn as snsimport randomfrom sklearn.metrics import accuracy_scorefrom sklearn.metrics import f1_scorefrom data import readfilefrom data import dealfile# 导入网络模型from unembed_transformer_net_d import *from render import *# 计算核心device = Noneif torch.cuda.is_available():    device = torch.device("cuda")elif torch.backends.mps.is_built():    device = torch.device("mps")else:    device = torch.device("cpu")print(device)# 数据读取# 源数据# pathfile_test01 = '../../../数据/19.FMCRD_Data/test_load0_1e_m15_200x5.csv'# pathfile_test02 = '../../../数据/19.FMCRD_Data/test_noisy_1e_m15_200x5HI.csv'# pathfile_test03 = '../../../数据/19.FMCRD_Data/test_noisy_1e_m15_200x5LO.csv'# pathfile_test04 = '../../../数据/19.FMCRD_Data/test_noisy_1e_m15_200x5MED.csv'# pathfile_train01 = '../../../数据/19.FMCRD_Data/train_load0_1e_m15_200x5.csv'# pathfile_train02 = '../../../数据/19.FMCRD_Data/train_noisy_1e_m15_200x5HI.csv'# pathfile_train03 = '../../../数据/19.FMCRD_Data/train_noisy_1e_m15_200x5LO.csv'# pathfile_train04 = '../../../数据/19.FMCRD_Data/train_noisy_1e_m15_200x5MED.csv'# undersampling10pathfile_source_true = '../../../数据/SG_T_YY/SG_source_true.csv'pathfile_source_false = '../../../数据/SG_T_YY/SG_source_false.csv'pathfile_target_true = '../../../数据/SG_T_YY/SG_target_true.csv'pathfile_target_false = '../../../数据/SG_T_YY/SG_target_false.csv'read01 = readfile(pathfile_source_true)read02 = readfile(pathfile_source_false)read03 = readfile(pathfile_target_true)read04 = readfile(pathfile_target_false)YY_data_source_true = read01.returndata()YY_data_source_false = read02.returndata()YY_data_target_true = read03.returndata()YY_data_target_false = read04.returndata()# # 选择最简单办法 源域替换到目标域# YY_data_source_true = YY_data_target_true# YY_data_source_false = YY_data_target_false# 选择最简单办法 从源域拿出1/10r = 0.1YY_data_source_true = YY_data_source_true.iloc[0:int(YY_data_source_true.shape[0]*r),:]YY_data_source_false = YY_data_source_false.iloc[0:int(YY_data_source_false.shape[0]*r),:]# 反向# YY_data_source_true = YY_data_target_false# YY_data_source_false = YY_data_target_true# print("origin_data_test01", origin_data_test01)# print("origin_data_test02", origin_data_test02)# print("origin_data_test03", origin_data_test03)# print("origin_data_test04", origin_data_test04)# print("origin_data_train01", origin_data_train01)# print("origin_data_train02", origin_data_train02)# print("origin_data_train03", origin_data_train03)# print("origin_data_train04", origin_data_train04)# 数据分析# from data import dealfile# deal01 = dealfile(origin_data_train01)# deal01.dealdata()# deal02 = dealfile(origin_data_test02)# deal02.dealdata()# deal03 = dealfile(origin_data_test03)# deal03.dealdata()# deal04 = dealfile(origin_data_test04)# deal04.dealdata()# 转化为numpy的array格式# train_info_01 = np.array(origin_data_train01.iloc[0:-1:100, 0:-1], dtype = 'float32')# train_info_02 = np.array(origin_data_train02.iloc[0:-1:100, 0:-1], dtype = 'float32')# train_info_03 = np.array(origin_data_train03.iloc[0:-1:100, 0:-1], dtype = 'float32')# train_info_04 = np.array(origin_data_train04.iloc[0:-1:100, 0:-1], dtype = 'float32')# 源域划分训练集和验证集# 0.85作为训练 0.15和异常(0.1 0.2 ~~~)作为验证eval_rate= 0.01false_userate = 0.01spilt_point01 = math.floor((1-eval_rate)*YY_data_source_true.shape[0])spilt_point02 = math.floor(false_userate*len(YY_data_source_false))# 训练数据YY_train_data = YY_data_source_true.iloc[0:spilt_point01, :-2]# 验证数据# 验证数据合并 0.15正常与异常一部分YY_eval_data_true = YY_data_source_true.iloc[spilt_point01:-1, :-2]YY_eval_data_false = YY_data_source_true.iloc[0:spilt_point02, :-2]YY_eval_data_true = YY_eval_data_true.reset_index(drop=True)YY_eval_data_false = YY_eval_data_false.reset_index(drop=True)YY_eval_data = pd.concat([YY_eval_data_true, YY_eval_data_false], axis=0, ignore_index=True)# 验证数据标签eval_label01 = np.zeros(len(YY_eval_data_true), dtype = 'long')eval_label02 = np.ones(len(YY_eval_data_false), dtype = 'long')YY_eval_label = np.append(eval_label01,eval_label02)# 无实际测试# 数据转化为numpyYY_train_data = np.array(YY_train_data, dtype='float32')YY_eval_data = np.array(YY_eval_data, dtype='float32')# 数据形成source target batchs# 批次化数据def batchify(data, bsz):    """    将数据映射为连续的数字 处理数据转化为nbatch的形式    """    nbatch = data.shape[0] // bsz    # 数据中除去多余余数部分    data = data[:nbatch * bsz, :]    # 使用view对data进行矩阵变化    data = data.reshape(nbatch, -1, data.shape[1], order='C')    # print(data.shape)    data = torch.tensor(data)    return data.to(device)def label_batchify(data, bsz):    """    将数据映射为连续的数字 处理数据转化为nbatch的形式    """    nbatch = data.shape[0] // bsz    # 数据中除去多余余数部分    data = data[:nbatch * bsz]    # 使用view对data进行矩阵变化    data = data.reshape(nbatch, -1, order='C')    data = data[:,0]    # print(data.shape)    data = torch.tensor(data)    return data.to(device)# 使用batch_ify来处理训练数据# 训练集bsztrain_batch_size = 100# 测试验证bszeval_batch_size = 100# 将训练 验证数据形成批次YY_train_data = batchify(YY_train_data, train_batch_size)YY_eval_data = batchify(YY_eval_data, eval_batch_size)YY_eval_label = label_batchify(YY_eval_label, eval_batch_size)# 时序长度允许最大值为35 1200 的因数bptt = 10def get_batch(source, i):    # 复制任务，不需要留一个了 没有-1    seq_len = min(bptt, len(source) - i)        # 语言模型训练的源数据的是将batchify的结果的切片[i:i+seq_len]    data = source[i:i+seq_len]        # 根据复制模型 目标就是输入    # 因为最后目标数据的切片会越界 因此使用view(-1)保证形状正常 array应该使用flat    target = data    return data, target# 设置模型超参数 初始化模型V = 20 # 1000 20hid_model = 1# 编码器层数量nlayers = 1 # 4# 词嵌入大小为200d_model = 13# 前馈全连接的节点数nhid = 20 # 2000 20# 多头注意力机制nhead = 1 # 7 2# 置0比率 复制模型 设为0.01dropout = 0.01model = make_model(V,hid_model,nlayers,d_model,nhid,nhead,train_batch_size,dropout)# 导入源域模型的训练参数state_dict = torch.load('./Net/Transformer_Testmachine_SG_source_detection_net_params.pkl', map_location=torch.device('cpu'))model.load_state_dict(state_dict)# 将模型迁移到gpumodel.to(device)# 模型初始化后 接下来损失函数与优化方法选择# 损失函数 # 我们使用nn自带的交叉熵损失 分类问题时采用# criterion = nn.CrossEntropyLoss()# MSELoss函数 计算输入和输出之差的平方 可输出序列标量或均方差criterion = nn.MSELoss()# 学习率初始值为5.0lr = 5# 优化器选择torch自带的SGD随机梯度下降方法 并把lr传入其中optimizer = torch.optim.SGD(model.parameters(), lr=lr)# 定义学习率调整器 使用torch自带的lr_scheduler 将优化器传入scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.99)# 定义训练轮数epochs = 80# 训练记录'''lr   gamma   endloss  epochs5    0.99    726      20005    0.999   669      20005    1       572      2000  波动10   0.99999  1000    2000  波动5    0.99999  401      5000 波动5    0.99     失败       50005    0.9999   319 367      50005    0.999    341       50005    0.999    449       200005    0.99    726      3000'''# 训练 验证 测试# 模型训练def train():    """    train model    """    # 模型开启训练模式    model.train()    # 初始损失定义为0    total_loss = 0    # 获得当前时间    start_time = time.time()    # 开始遍历批次数据    for batch, i in enumerate(range(0, YY_train_data.shape[0] - 1, bptt)):        # 通过get_batch        data, targets = get_batch(YY_train_data, i)        # 获得mask        source_mask = Variable(torch.ones(data.shape[1])).to(device)        # source_mask = subsequent_mask(data.shape[1]).to(device)        target_mask = subsequent_mask(targets.shape[1]).to(device)        # 设置优化器初始为0梯度        optimizer.zero_grad()                # 将数据装入model得到输出        output, hid_output = model(data, data, source_mask, target_mask)        # 将输入与目标传入损失函数对象        # print(output.shape)        # print(targets.shape)        # print(output)        # print(targets)                loss = criterion(output.reshape(-1), targets.reshape(-1))        # print(loss.shape, loss)                # 损失进行反向传播得到损失总和        loss.backward()        # 使用nn自带的clip_grad_norm_进行梯度规范化 防止出现梯度消失或者爆炸        nn.utils.clip_grad_norm_(model.parameters(), 0.5)        # 模型参数更新        optimizer.step()        # 损失累加得到总损失        total_loss += loss.item()        # 日志打印间隔为        log_interval = 10        if batch % log_interval == 0 and batch >= 0:            # 平均损失为 总损失 / log_interval            cur_loss = total_loss / log_interval            # 需要时间 当前时间 - 起始时间            elapsed = time.time() - start_time            # 打印 轮数 当前批次 总批次 当前学习率 训练速度(每毫秒处理多少批次)            # 平均损失 以及困惑度 困惑度是语言模型的重要标准 计算方法            # 交叉熵平均损失取自然对数的底数            print('| epoch {:3d} | {:5d}/{:5d} batches | '                  'lr {:02.8f} | ms/batch {:5.2f} | '                  'loss {:5.2f} | mape {:8.2f}'.format(                      epoch, batch, len(YY_train_data) // bptt,                       scheduler.get_lr()[0], elapsed * 1000 / log_interval,                      cur_loss, cur_loss))                    # 每个批次结束后总损失归0            total_loss = 0            # 开始时间取当前时间            start_time = time.time()# 模型评估def evaluate(eval_model, data_source):    """    评估函数 包括模型验证和测试    Parameters    ----------    eval_model : model的对象        DESCRIPTION.        每轮训练后或验证后的模型    data_source : dataset        DESCRIPTION.        验证或测试数据集    Returns    -------    total_loss : TYPE int        DESCRIPTION.        总损失    """    # 模型进入评估模式    eval_model.eval()    # 总损失    total_loss = 0    with torch.no_grad():        # 与训练步骤基本一致        for i in range(0, data_source.size(0) - 1, bptt):            data, targets = get_batch(data_source, i)            if targets.shape[0] < bptt:                break;            # 获得mask            source_mask = Variable(torch.ones(data.shape[1])).to(device)            # source_mask = subsequent_mask(data.shape[1]).to(device)            target_mask = subsequent_mask(data.shape[1]).to(device)            # 设置优化器初始为0梯度            output, hid_output = eval_model(data, data, source_mask, target_mask)                        # 计算均方差损失            total_loss += criterion(output.reshape(-1).cpu(), targets.reshape(-1).cpu())            # print('output', output)            # print('targets', targets)                        # print('total', total)            # print('correct', correct)            # print('predicted', predicted)                # 返回每轮总损失    return total_loss# 模型验证评估# 首先初始化最佳验证损失 初始值为无穷大best_train_loss = float("inf")# 定义最佳模型训练变量 初始值为Nonebest_model = model# 损失记录lossmem = [[],[]]# 使用for循环遍历轮数for epoch in range(1, epochs+1):    # 首先获得轮数开始时间    epoch_start_time = time.time()    # 调用训练函数    train()    # 该轮训练后我们的模型参数已经发生了变化        # # train_loss    # train_loss = evaluate(model, YY_train_data)    # # val_loss = 0    # # 之后打印每轮的评估日志 分别有轮数 耗时 验证损失 验证困惑度    # print('-' * 89)    # print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | '    #       'train mape {:2.2f}'.format(epoch, (time.time() - epoch_start_time),     #                                  train_loss, train_loss))    # print('-' * 89)        # 将模型和评估数据传入评估函数    eval_loss = evaluate(model, YY_eval_data)    lossmem[0].append(eval_loss)            # val_loss = 0    # 之后打印每轮的评估日志 分别有轮数 耗时 验证损失 验证困惑度    print('-' * 89)    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '          'valid mape {:2.2f}'.format(epoch, (time.time() - epoch_start_time),                                       eval_loss, eval_loss))    print('-' * 89)        # 我们将比较哪一轮损失最小 赋值给best_val_loss,    # 并取该损失下模型的best_model    if eval_loss < best_train_loss:        best_train_loss = eval_loss        best_model= model    # 每轮都会对优化方法的学习率做调整    scheduler.step()# 绘制损失下降图# 绘制好看一点my_font=font_manager.FontProperties(fname='/Users/mayuan/WorkSpace/Science/7毕业论文/workspace/图表/仿宋_常规.ttf',size='large')plt.figure(figsize=(8, 2), dpi=400)plt.plot(lossmem[0])# plt.ylim(0, 6000000)plt.xlabel('迭代次数', rotation=0, fontproperties=my_font)plt.ylabel('损失', rotation=90, fontproperties=my_font)plt.title('电子伺服系统丝杠磨损异常监测目标域正常模式自监督训练', rotation=0, fontproperties=my_font)plt.show()# # 模型保存# torch.save(best_model, './Net/Transformer_Testmachine_SG_target_detection_net.pkl')# # 只保存神经网络的模型参数# torch.save(best_model.state_dict(), './Net/Transformer_Testmachine_SG_target_detection_net_params.pkl')"""# 模型输出特征分布"""# 输入验证数据 带有验证标签# 定义其重构误差这里采用平方均值来表示误差，并将异常和正常数据的误差密度可视化def get_recon_err(X):    # 模型验证模式    model.eval()    # 记录输出的err    all_err = np.array([])    # 得到输出    with torch.no_grad():        for i in range(0, X.size(0), bptt):            # 通过get_batch            data, targets = get_batch(X, i)            # 获得mask            source_mask = Variable(torch.ones(data.shape[1])).to(device)            # source_mask = subsequent_mask(data.shape[1]).to(device)            target_mask = subsequent_mask(targets.shape[1]).to(device)                        output, hid_output = model(data, data, source_mask, target_mask)                        # print('output', output.shape)            # print('targets', targets.shape)                        err = torch.mean((output - targets) ** 2, dim=[1,2]).detach().cpu().numpy()                        # print(err.shape)            # print(all_err.shape)            if not all_err.any():                all_err = err            else:                all_err = np.append(all_err, err, axis=0)                return all_err# train是B*T*D; 得到B*1 get_recon_err的作用recon_err_test = get_recon_err(YY_eval_data)YY_eval_label_cpu = YY_eval_label.cpu()recon_err = recon_err_testlabels = np.concatenate([np.zeros(YY_eval_data_true.shape[0] // train_batch_size),                         np.ones(YY_eval_data.shape[0] - YY_eval_data_true.shape[0] // train_batch_size)])index = np.arange(0, len(labels))"""# 部分绘图"""plt.figure(figsize=(8, 2), dpi=400)plt.xlabel('损失值', rotation=0, fontproperties=my_font)plt.ylabel('稠密度', rotation=90, fontproperties=my_font)plt.title('电子伺服系统丝杠磨损异常监测目标域正常与异常数量在损失值上的分布情况', rotation=0, fontproperties=my_font)sns.kdeplot(recon_err[labels == 0], fill=True, color='green', label='Noramal')sns.kdeplot(recon_err[labels == 1], fill=True, color='red', label='Anomaly')# plt.xlim(-25000,200000)plt.legend()plt.show()# 最后设置不同阈值，寻找最佳的划定阈值。threshold = np.linspace(0, 100000, 1000)# 准确率 精确率召回率调和平均值 acc_list = []f1_list = []for t in threshold:    y_pred = (recon_err_test > t).astype(int)    # print(y_pred.shape)    # print(YY_eval_label.shape)    acc_list.append(accuracy_score(YY_eval_label_cpu, y_pred))    f1_list.append(f1_score(YY_eval_label_cpu, y_pred))plt.figure(figsize=(8, 6), dpi=400)plt.plot(threshold, acc_list, c='y', label='Acc')plt.plot(threshold, f1_list, c='b', label='F1') # 精确率和召回率的调和平均# plt.xlim(-2000,110000)plt.xlabel('电子伺服系统目标域重构模型的损失阈值', rotation=0, fontproperties=my_font)plt.ylabel('异常监测效果（%）', rotation=90, fontproperties=my_font)plt.legend()plt.show()i = np.argmax(f1_list)t = threshold[i]score = f1_list[i]print('threshold: %.3f,  f1 score: %.3f' % (t, score))y_pred = (recon_err_test > t).astype(int)# 混淆矩阵（可以封装成一个包，后续导入）import numpy as npimport pandas as pdimport matplotlib.pyplot as pltfrom sklearn.metrics import confusion_matrix, roc_auc_scoreimport seaborn as snsfrom sklearn.preprocessing import MinMaxScaler# 最后编写绘制画混淆矩阵的函数，可以调用绘制出异常检测的混淆矩阵以及异常和正常点的分类。class visualization:    labels = ["Noramal", "Anomaly"]    def draw_confusion_matrix(self, y, ypred):        matrix = confusion_matrix(y, ypred)        plt.figure(figsize=(10, 8), dpi=400)        colors = ["#b5d9dc", "blue"]        # sns.heatmap(matrix, xticklabels=self.labels, yticklabels=self.labels, cmap=colors, annot=True, fmt="d")        sns.heatmap(matrix, xticklabels=self.labels, yticklabels=self.labels, annot=True, cmap='Blues', fmt='g', cbar=True, square=True)        plt.title("混淆矩阵", rotation=0, fontproperties=my_font)        plt.ylabel('真实标签', rotation=90, fontproperties=my_font)        plt.xlabel('监测标签', rotation=0, fontproperties=my_font)        plt.show()    def draw_anomaly(self, y, error, threshold):        groupSDF = pd.DataFrame({'error': error,                                 'true': y}).groupby('true')        figure, axes = plt.subplots(figsize=(12, 8), dpi=400)        for name, group in groupSDF:            axes.plot(group.index, group.error, marker='x' if name == 1 else 'o', linestyle='',                      color='r' if name == 1 else 'g', label="Anomaly" if name == 1 else "Normal")        axes.hlines(threshold, axes.get_xlim()[0], axes.get_xlim()[1], colors='b', zorder=100, label="Threshold")        # axes.legend()        plt.title("最佳阈值下的区分度", fontproperties=my_font)        plt.ylabel("损失值", fontproperties=my_font)        plt.xlabel("验证样本", fontproperties=my_font)        # plt.ylim(-2000, 110000)        plt.show()    def draw_error(self, error, threshold):        plt.plot(error, marker='o', ms=3.5, linestyle='', label='Point')        plt.hlines(threshold, xmin=0, xmax=len(error) - 1, colors='b', zorder=100, label='Threshold')        plt.legend()        plt.title("Reconstruction error")        plt.ylabel("Error")        plt.xlabel("Data")        plt.show()viz = visualization()viz.draw_confusion_matrix(YY_eval_label_cpu,y_pred)viz.draw_anomaly(YY_eval_label_cpu,recon_err_test,threshold[i])# 由于数据集前面部分都是正常标签值，最后顺序的为异常值，现在将其排序打乱并重新可视化。c = list(zip(YY_eval_label_cpu, y_pred, recon_err_test))random.Random(100).shuffle(c)YY_eval_label_cpu, y_pred, recon_err_test = zip(*c)viz.draw_anomaly(YY_eval_label_cpu,recon_err_test,threshold[i])