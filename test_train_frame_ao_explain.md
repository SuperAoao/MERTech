# `test_train_frame_ao` 逐行说明

对照文件：`test_train_frame_ao`。每一行：**先写涉及 API / 机制**，**再写在本脚本里的具体作用**。  
文末附有与当前脚本一致的**完整源码**（普通代码块，**没有**用 `#` 把每行注释掉）。运行实验请执行 `test_train_frame_ao`，不要把这个 Markdown 当 Python 执行。

---

### L1 `import sys`

- **API**：Python 的 `import` 把模块加载进当前命名空间。
- **含义**：后续要用 `sys.path` 把 `function` 目录加入模块搜索路径。

### L2 `import matplotlib`

- **API**：导入 `matplotlib` 包（绑图、后端等）。
- **含义**：下一行会设置非交互后端，避免无 GUI 环境报错。

### L3 `matplotlib.use('Agg')`

- **API**：`matplotlib.use(backend)` 须在创建 `Figure` 之前调用，用于指定渲染后端。
- **含义**：`Agg` 为纯图像后端，适合服务器或无显示器场景。

### L4 `import datetime`

- **API**：标准库 `datetime` 提供日期时间类型与 `datetime.datetime.now()` 等。
- **含义**：为下一行取「当前时间」提供模块。

### L5 `date = datetime.datetime.now()`

- **API**：`datetime.datetime.now()` 返回当前本地时间的 `datetime` 对象。
- **含义**：保存运行时刻；本脚本后续未再使用 `date`（若你扩展日志/路径可复用）。

### L6 `sys.path.append('./function')`

- **API**：`sys.path` 是模块搜索路径列表，`list.append` 在末尾追加一项。
- **含义**：让解释器能找到项目下的 `function` 包，从而 `from function.xxx` 生效。

### L7 `from function.model import *`

- **API**：`from pkg import *` 按包定义批量导入符号（取决于 `__all__` 与模块导出）。
- **含义**：引入 `SSLNet` 等模型定义。

### L8 `from function.lib import *`

- **API**：同上。
- **含义**：引入 `compute_metrics_with_note`、数据集类等与评测/损失相关的符号。

### L9 `from function.load_data import *`

- **API**：同上。
- **含义**：引入 `load`，用于读取 wav/csv 并做测试集分块。

### L10 `import numpy as np`

- **API**：`import ... as` 为模块起别名。
- **含义**：后面用 `np.concatenate`、数组运算等。

### L11 `import random`

- **API**：标准库伪随机数，`seed` 可固定序列。
- **含义**：在 `get_random_seed` 里与 `torch`/`numpy` 一起固定随机性。

### L12 `import os`

- **API**：操作系统接口（环境变量、路径等）。
- **含义**：设置 `PYTHONHASHSEED` 等以提升可复现性。

### L13 `import torch`

- **API**：PyTorch 主包：张量、`nn.Module`、`device`、`manual_seed` 等。
- **含义**：加载模型、推理、设备与随机种子。

### L14 `import torch.nn.functional as F`

- **API**：函数式接口（不封装成 `nn.Module` 的算子）。
- **含义**：使用 `F.sigmoid` 把 logits 转为概率（也可用 `torch.sigmoid`）。

### L15 `from transformers import Wav2Vec2FeatureExtractor`

- **API**：HuggingFace 中与 Wav2Vec2 系模型配套的波形预处理类。
- **含义**：与预训练一致地把原始波形变成模型所需的 `input_values`。

### L17 `def start_test():`

- **API**：`def` 定义可调用函数。
- **含义**：把整段测试流程封装成 `start_test`，文件末尾再调用。

### L18 `def get_random_seed(seed):`

- **API**：嵌套函数定义；可访问外层作用域变量。
- **含义**：局部封装「一次性设完各库种子」的逻辑。

### L19 `random.seed(seed)`

- **API**：初始化 Python `random` 模块的全局 RNG。
- **含义**：固定标准库随机行为。

### L20 `os.environ['PYTHONHASHSEED'] = str(seed)`

- **API**：进程环境变量读写。
- **含义**：减轻 `dict`/`set` 迭代顺序等哈希随机性，利于复现。

### L21 `np.random.seed(seed)`

- **API**：NumPy 旧版全局随机种子接口（行为随版本略有差异）。
- **含义**：固定 NumPy 随机数。

### L22 `torch.manual_seed(seed)`

- **API**：设置 PyTorch 在 CPU 上的默认随机数生成器。
- **含义**：固定 PyTorch 在 CPU 上的随机性。

### L23 `torch.cuda.manual_seed(seed)`

- **API**：为当前 GPU 设置随机种子。
- **含义**：在 CUDA 可用时固定 GPU 随机性。

### L24 `torch.backends.cudnn.deterministic = True`

- **API**：cuDNN 是否选用确定性实现。
- **含义**：减少 GPU 上非确定性（可能略慢）。

### L25 `torch.backends.cudnn.benchmark = False`

- **API**：`benchmark=True` 时 cuDNN 会对卷积等做 autotune。
- **含义**：关闭 autotune，配合 deterministic 更利于复现。

### L26 `get_random_seed(42)`

- **API**：函数调用。
- **含义**：用种子 `42` 初始化上述所有随机源。

### L28 注释 `#load model`

- **API**：`#` 单行注释，解释器忽略。
- **含义**：标记下面开始加载网络权重。

### L29 `model = SSLNet(...).to(device)`

- **API**：`nn.Module` 子类实例化；`.to(device)` 把参数/缓冲区搬到指定设备。
- **含义**：按 `config` 的 `URL`、`NUM_LABELS`、`MIN_MIDI`、`MAX_MIDI` 等构建与训练一致的结构，并放到 `cuda:0` 或 `cpu`。

### L30 注释掉的 `torch.load(...)`

- **API**：整行以 `#` 注释，不执行。
- **含义**：保留旧 checkpoint 路径示例。

### L31 `state_dict = torch.load(..., map_location="cpu")`

- **API**：`torch.load` 反序列化；`map_location` 指定张量映射到的设备。
- **含义**：从磁盘读取训练保存的 `state_dict`；先映到 CPU 可避免无 GPU 时加载失败。

### L32 `model.load_state_dict(state_dict)`

- **API**：`Module.load_state_dict` 按键名把张量写入模型参数（默认 `strict=True`）。
- **含义**：把权重载入 `model`，要求与当前 `SSLNet` 结构一致。

### L34 `print('finishing loading model')`

- **API**：内置 `print` 写标准输出。
- **含义**：提示模型已加载。

### L36–L37 `wav_dir` / `csv_dir`

- **API**：字符串 `+` 拼接。
- **含义**：`DATASET` 来自 `config`，拼出测试音频目录与标签 CSV 目录。

### L38 `test_group = ['test']`

- **API**：列表字面量。
- **含义**：告诉 `load` 只读 `test` 子目录（与 `load_data.py` 分支一致）。

### L39 `Xte, Yte, ... = load(...)`

- **API**：多返回值解包；`load` 为项目自定义函数。
- **含义**：读测试集；`Xte` 为波形块列表，`Yte`/`Yte_p`/`Yte_o` 为 IPT/pitch/onset 标签块列表；后两个 `_` 忽略 avg/std。

### L40 `print('finishing loading dataset')`

- **API**：`print`。
- **含义**：提示数据读完。

### L42 `processor = Wav2Vec2FeatureExtractor.from_pretrained(...)`

- **API**：`from_pretrained` 从 Hub 或缓存加载配置与预处理；`trust_remote_code=True` 允许仓库自定义代码。
- **含义**：与训练使用同一套波形预处理。

### L44 注释 `# start predict`

- **API**：注释。
- **含义**：标记进入推理。

### L45 `print('start predicting...')`

- **API**：`print`。
- **含义**：提示开始逐块预测。

### L46 `model.eval()`

- **API**：`nn.Module.eval()` 切换到评估模式（如关闭 dropout）。
- **含义**：推理前固定网络行为。

### L47 `ds = 0`

- **API**：整数赋值。
- **含义**：标志位：第一次循环初始化累积数组，否则沿时间维拼接。

### L48 `for i, x in enumerate(Xte):`

- **API**：`enumerate` 产生 `(索引, 元素)`；`Xte` 可迭代。
- **含义**：对每个测试音频块 `x` 单独前向，避免整曲一次占满显存。

### L49 `data = processor(...)["input_values"].float().to(device)`

- **API**：`processor` 返回字典；`["input_values"]` 取波形张量；`.float()` 转 dtype；`.to(device)` 搬设备。
- **含义**：把当前块变成模型输入张量（通常带 batch 维）。

### L50–L52 `target` / `target_p` / `target_o`

- **API**：列表按索引取下标 `i`。
- **含义**：当前块对应的 IPT、pitch、onset 真值（形状分别约为 `(7, T)`、`(N_pitch, T)`、`(1, T)`）。

### L53 `IPT_pred, pitch_pred, onset_pred = model(data)`

- **API**：`nn.Module.__call__` 触发 `forward`。
- **含义**：得到三个任务的 logits，形状大致为 `[1, C, T']`。

### L54–L56 `f_pred` / `p_pred` / `o_pred`

- **API**：`Tensor.squeeze(0)` 去掉大小为 1 的第 0 维；`F.sigmoid` 逐元素 sigmoid；`.cpu().numpy()` 转 NumPy（`.data` 为旧式取张量）。
- **含义**：去掉 batch 维，得到概率图并放到 CPU，便于 `numpy` 拼接。

### L57–L59 按 `target.shape[-1]` 截断

- **API**：NumPy 切片 `[:, :L]`。
- **含义**：若模型输出帧数与标签帧数不一致，按标签长度截断预测（不在这里补零）。

### L60–L68 首块初始化

- **API**：`if`；多重赋值；`+=`。
- **含义**：第一块直接赋给 `all_tar`、`all_pred` 等累积变量，并把 `ds` 置非零。

### L69–L75 `else` 分支拼接

- **API**：`np.concatenate(..., axis=-1)` 在最后一维拼接。
- **含义**：把后续各块沿**时间维**接到整曲长度。

### L76 `threshold = 0.5`

- **API**：浮点字面量。
- **含义**：帧级二值化阈值。

### L77–L79 IPT 二值化

- **API**：变量引用同一数组；布尔掩码赋值。
- **含义**：`pred_IPT` 先指向 `all_pred`，再把概率变成 0/1。

### L80–L82、L83–L85 pitch / onset 二值化

- **API**：同上。
- **含义**：对 pitch、onset 概率做同样阈值化。

### L86–L88 别名

- **API**：名称绑定到同一对象。
- **含义**：`target_IPT`/`tar_pitch`/`tar_onset` 指向整曲真值，供指标函数使用。

### L91 `compute_metrics_with_note_no_infer(...)`

- **API**：项目自定义函数，返回 `(metrics_dict, _)`。
- **含义**：**不做** onset 后处理时的 IPT 帧级 / note 级指标。

### L92 `compute_metrics_with_note(...)`

- **API**：同上。
- **含义**：用**预测 onset** 对 IPT 做后处理后再算指标；`roll` 为中间结果，此处未再用。

### L93–L108 `print` 指标

- **API**：`print`；字典用键取值。
- **含义**：分别打印后处理前、后的 precision/recall/F1/accuracy 等。

### L110–L116 按 IPT 类分别算指标

- **API**：`ndarray[i, :]` 取第 `i` 行；`reshape((1, -1))` 变成二维 `1×T`；`compute_metrics_with_note` 内部会做 `transpose` 等与 `mir_eval` 对接。
- **含义**：对 IPT 类 0…6 各算一遍「单类 IPT 卷帘 + 全曲 onset」的指标。

**重要**：`tar_onset` 的形状是 **`(1, T)`**（onset 只有**一行**）。  
L110 使用 `tar_onset[0, :]` 正确；**L111–L116 若写 `tar_onset[1, :]` … `tar_onset[6, :]` 会在运行时 `IndexError`**（第 0 维长度仅为 1）。正确做法是：先设 `onset_row = tar_onset[0, :].reshape((1, -1))`，七个 `compute_metrics_with_note` 的最后一个参数**都传 `onset_row`**。

### L118–L131 按技法名打印 F1

- **API**：`print`；字典索引。
- **含义**：把 `metrics0`…`metrics6` 对应到注释中的技法名（颤音、拨弦等）并打印 frame/note F1。

### L132–L135 `macro_frame_f1`

- **API**：浮点加减除；`list[0]` 取单元素列表里的标量。
- **含义**：七个 IPT 类的 frame F1 **宏平均**。

### L136–L139 `macro_note_f1`

- **API**：同上。
- **含义**：七个 IPT 类的 note F1 宏平均。

### L140–L141 打印宏平均

- **API**：`print`。
- **含义**：输出两条宏平均结果。

### L143 `start_test()`

- **API**：顶层函数调用；作为脚本入口时执行测试流程。
- **含义**：运行整套加载—推理—评测。

---

## 完整源码（与 `test_train_frame_ao` 一致，未加行首 `#`）

```python
import sys
import matplotlib
matplotlib.use('Agg')
import datetime
date = datetime.datetime.now()
sys.path.append('./function')
from function.model import *
from function.lib import *
from function.load_data import *
import numpy as np
import random
import os
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor

def start_test():
    def get_random_seed(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    get_random_seed(42)

    #load model
    model = SSLNet(url=URL, class_num=NUM_LABELS*(MAX_MIDI-MIN_MIDI+1), weight_sum=1, freeze_all=FREEZE_ALL).to(device)
    #state_dict = torch.load('data/model/mul_onset_share_weight_detach_ft(final)/mul_onset_share_weight_detach_ft-MERT-v1-95M/best_e_1970',map_location="cpu")
    state_dict = torch.load('data/model/baseline/best_e_2225',map_location="cpu")
    model.load_state_dict(state_dict)

    print('finishing loading model')

    wav_dir = DATASET + '/data'
    csv_dir = DATASET + '/labels'
    test_group = ['test']
    Xte, Yte, Yte_p, Yte_o,  _, _ = load(wav_dir, csv_dir, test_group, None, None)
    print ('finishing loading dataset')

    processor = Wav2Vec2FeatureExtractor.from_pretrained(URL, trust_remote_code=True)

    # start predict
    print('start predicting...')
    model.eval()
    ds = 0
    for i, x in enumerate(Xte):
        data = processor(x, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")["input_values"].float().to(device)
        target = Yte[i]
        target_p = Yte_p[i]
        target_o = Yte_o[i]
        IPT_pred, pitch_pred, onset_pred = model(data)
        f_pred = F.sigmoid(IPT_pred.squeeze(0)).data.cpu().numpy()
        p_pred = F.sigmoid(pitch_pred.squeeze(0)).data.cpu().numpy()
        o_pred = F.sigmoid(onset_pred.squeeze(0)).data.cpu().numpy()

        f_pred = f_pred[:, :target.shape[-1]]
        p_pred = p_pred[:, :target.shape[-1]]
        o_pred = o_pred[:, :target.shape[-1]]
        if ds == 0:
            all_tar = target
            all_pred = f_pred
            pitch_tar = target_p
            pp_pred = p_pred
            onset_tar = target_o
            oo_pred = o_pred
            ds += 1
        else:
            all_tar = np.concatenate([all_tar, target], axis=-1)
            all_pred = np.concatenate([all_pred, f_pred], axis=-1)
            pitch_tar = np.concatenate([pitch_tar, target_p], axis=-1)
            pp_pred = np.concatenate([pp_pred, p_pred], axis=-1)
            onset_tar = np.concatenate([onset_tar, target_o], axis=-1)
            oo_pred = np.concatenate([oo_pred, o_pred], axis=-1)
    threshold = 0.5
    pred_IPT = all_pred
    pred_IPT[pred_IPT > threshold] = 1
    pred_IPT[pred_IPT <= threshold] = 0
    pred_pitch = pp_pred
    pred_pitch[pred_pitch > threshold] = 1
    pred_pitch[pred_pitch <= threshold] = 0
    pred_onset = oo_pred
    pred_onset[pred_onset > threshold] = 1
    pred_onset[pred_onset <= threshold] = 0
    target_IPT = all_tar
    tar_pitch = pitch_tar
    tar_onset = onset_tar

    # compute metrics
    metrics_no_infer, _ = compute_metrics_with_note_no_infer(pred_IPT, target_IPT,pred_onset, tar_onset)
    metrics, roll = compute_metrics_with_note(pred_IPT, target_IPT, pred_onset, tar_onset)
    print("The result before post-processing：")
    print("IPT_frame_precision:", metrics_no_infer['metric/IPT_frame/precision'])
    print("IPT_frame_recall:", metrics_no_infer['metric/IPT_frame/recall'])
    print("IPT_frame_f1:", metrics_no_infer['metric/IPT_frame/f1'])
    print("IPT_frame_accuracy:", metrics_no_infer['metric/IPT_frame/accuracy'])
    print("IPT_note_precision:", metrics_no_infer['metric/note/precision'])
    print("IPT_note_recall:", metrics_no_infer['metric/note/recall'])
    print("IPT_note_f1:", metrics_no_infer['metric/note/f1'])
    print("The result after post-processing：")
    print("IPT_frame_precision:", metrics['metric/IPT_frame/precision'])
    print("IPT_frame_recall:", metrics['metric/IPT_frame/recall'])
    print("IPT_frame_f1:", metrics['metric/IPT_frame/f1'])
    print("IPT_frame_accuracy:", metrics['metric/IPT_frame/accuracy'])
    print("IPT_note_precision:", metrics['metric/note/precision'])
    print("IPT_note_recall:", metrics['metric/note/recall'])
    print("IPT_note_f1:", metrics['metric/note/f1'])

    metrics0, _ = compute_metrics_with_note(pred_IPT[0, :].reshape((1, -1)), target_IPT[0, :].reshape((1, -1)), pred_onset, tar_onset[0, :].reshape((1, -1)))
    metrics1, _ = compute_metrics_with_note(pred_IPT[1, :].reshape((1, -1)), target_IPT[1, :].reshape((1, -1)), pred_onset, tar_onset[1, :].reshape((1, -1)))
    metrics2, _ = compute_metrics_with_note(pred_IPT[2, :].reshape((1, -1)), target_IPT[2, :].reshape((1, -1)), pred_onset, tar_onset[2, :].reshape((1, -1)))
    metrics3, _ = compute_metrics_with_note(pred_IPT[3, :].reshape((1, -1)), target_IPT[3, :].reshape((1, -1)), pred_onset, tar_onset[3, :].reshape((1, -1)))
    metrics4, _ = compute_metrics_with_note(pred_IPT[4, :].reshape((1, -1)), target_IPT[4, :].reshape((1, -1)), pred_onset, tar_onset[4, :].reshape((1, -1)))
    metrics5, _ = compute_metrics_with_note(pred_IPT[5, :].reshape((1, -1)), target_IPT[5, :].reshape((1, -1)), pred_onset, tar_onset[5, :].reshape((1, -1)))
    metrics6, _ = compute_metrics_with_note(pred_IPT[6, :].reshape((1, -1)), target_IPT[6, :].reshape((1, -1)), pred_onset, tar_onset[6, :].reshape((1, -1)))

    print("vibrato_frame_f1:", metrics0['metric/IPT_frame/f1'])
    print("vibrato_note_f1:", metrics0['metric/note/f1'])
    print("plucks_frame_f1:", metrics1['metric/IPT_frame/f1'])
    print("plucks_note_f1:", metrics1['metric/note/f1'])
    print("UP_frame_f1:", metrics2['metric/IPT_frame/f1'])
    print("UP_note_f1:", metrics2['metric/note/f1'])
    print("DP_frame_f1:", metrics3['metric/IPT_frame/f1'])
    print("DP_note_f1:", metrics3['metric/note/f1'])
    print("glissando_frame_f1:", metrics4['metric/IPT_frame/f1'])
    print("glissando_note_f1:", metrics4['metric/note/f1'])
    print("tremolo_frame_f1:", metrics5['metric/IPT_frame/f1'])
    print("tremolo_note_f1:", metrics5['metric/note/f1'])
    print("PN_frame_f1:", metrics6['metric/IPT_frame/f1'])
    print("PN_note_f1:", metrics6['metric/note/f1'])
    macro_frame_f1 = float(metrics0['metric/IPT_frame/f1'][0] + metrics1['metric/IPT_frame/f1'][0] +
                      metrics2['metric/IPT_frame/f1'][0] + metrics3['metric/IPT_frame/f1'][0] +
                      metrics4['metric/IPT_frame/f1'][0] + metrics5['metric/IPT_frame/f1'][0] +
                      metrics6['metric/IPT_frame/f1'][0])/7.0
    macro_note_f1 = float(metrics0['metric/note/f1'][0] + metrics1['metric/note/f1'][0] +
                     metrics2['metric/note/f1'][0] +metrics3['metric/note/f1'][0] +
                     metrics4['metric/note/f1'][0] +metrics5['metric/note/f1'][0] +
                     metrics6['metric/note/f1'][0])/7.0
    print("macro_frame_f1:",macro_frame_f1)
    print("macro_note_f1:",macro_note_f1)

start_test()
```
