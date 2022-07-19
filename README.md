## 4. 轻定制功能

对于简单的抽取目标可以直接使用```paddlenlp.Taskflow```实现零样本（zero-shot）抽取，对于细分场景我们推荐使用轻定制功能（标注少量数据进行模型微调）以进一步提升效果。下面通过`报销工单信息抽取`的例子展示如何通过5条训练数据进行UIE模型微调。

#### 代码结构

```shell
.
├── utils.py          # 数据处理工具
├── model.py          # 模型组网脚本
├── doccano.py        # 数据标注脚本
├── doccano.md        # 数据标注文档
├── finetune.py       # 模型微调脚本
├── evaluate.py       # 模型评估脚本
└── README.md
```

#### 数据标注

我们推荐使用数据标注平台[doccano](https://github.com/doccano/doccano) 进行数据标注，本示例也打通了从标注到训练的通道，即doccano导出数据后可通过[doccano.py](./doccano.py)脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。

原始数据示例：

```text
深大到双龙28块钱4月24号交通费
```

抽取的目标(schema)为：

```python
schema = ['出发地', '目的地', '费用', '时间']
```

标注步骤如下：

- 在doccano平台上，创建一个类型为``序列标注``的标注项目。
- 定义实体标签类别，上例中需要定义的实体标签有``出发地``、``目的地``、``费用``和``时间``。
- 使用以上定义的标签开始标注数据，下面展示了一个doccano标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167336891-afef1ad5-8777-456d-805b-9c65d9014b80.png height=100 hspace='10'/>
</div>

- 标注完成后，在doccano平台上导出文件，并将其重命名为``doccano_ext.json``后，放入``./data``目录下。

- 这里我们提供预先标注好的文件[doccano_ext.json](https://bj.bcebos.com/paddlenlp/datasets/uie/doccano_ext.json)，可直接下载并放入`./data`目录。执行以下脚本进行数据转换，执行后会在`./data`目录下生成训练/验证/测试集文件。

```shell
python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --task_type ext \
    --save_dir ./data \
    --splits 0.8 0.2 0
```


可配置参数说明：

- ``doccano_file``: 从doccano导出的数据标注文件。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``negative_ratio``: 最大负例比例，该参数只对抽取类型任务有效，适当构造负例可提升模型效果。负例数量和实际的标签数量有关，最大负例数量 = negative_ratio * 正例数量。该参数只对训练集有效，默认为5。为了保证评估指标的准确性，验证集和测试集默认构造全负例。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``task_type``: 选择任务类型，可选有抽取和分类两种类型的任务。
- ``options``: 指定分类任务的类别标签，该参数只对分类类型任务有效。默认为["正向", "负向"]。
- ``prompt_prefix``: 声明分类任务的prompt前缀信息，该参数只对分类类型任务有效。默认为"情感倾向"。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为True。
- ``seed``: 随机种子，默认为1000.
- ``separator``: 实体类别/评价维度与分类标签的分隔符，该参数只对实体/评价维度级分类任务有效。默认为"##"。

备注：
- 默认情况下 [doccano.py](./doccano.py) 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 [doccano.py](./doccano.py) 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。
- 对于从doccano导出的文件，默认文件中的每条数据都是经过人工正确标注的。

更多**不同类型任务（关系抽取、事件抽取、评价观点抽取等）的标注规则及参数说明**，请参考[doccano数据标注指南](doccano.md)。

#### 模型微调

单卡启动：

```shell
python finetune.py \
    --train_path ./data/train.txt \
    --dev_path ./data/dev.txt \
    --save_dir ./checkpoint \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --num_epochs 100 \
    --model uie-base \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device gpu
```

多卡启动：

```shell
python -u -m paddle.distributed.launch --gpus "0,1" finetune.py \
  --train_path ./data/train.txt \
  --dev_path ./data/dev.txt \
  --save_dir ./checkpoint \
  --learning_rate 1e-5 \
  --batch_size 16 \
  --max_seq_len 512 \
  --num_epochs 100 \
  --model uie-base \
  --seed 1000 \
  --logging_steps 10 \
  --valid_steps 100 \
  --device gpu
```

可配置参数说明：

- `train_path`: 训练集文件路径。
- `dev_path`: 验证集文件路径。
- `save_dir`: 模型存储路径，默认为`./checkpoint`。
- `learning_rate`: 学习率，默认为1e-5。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `num_epochs`: 训练轮数，默认为100。
- `model`: 选择模型，程序会基于选择的模型进行模型微调，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`和`uie-nano`，默认为`uie-base`。
- `seed`: 随机种子，默认为1000.
- `logging_steps`: 日志打印的间隔steps数，默认10。
- `valid_steps`: evaluate的间隔steps数，默认100。
- `device`: 选用什么设备进行训练，可选cpu或gpu。

#### 模型评估

通过运行以下命令进行模型评估：

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --batch_size 16 \
    --max_seq_len 512
```

评估方式说明：采用单阶段评价的方式，即关系抽取、事件抽取等需要分阶段预测的任务对每一阶段的预测结果进行分别评价。验证/测试集默认会利用同一层级的所有标签来构造出全部负例。

可开启`debug`模式对每个正例类别分别进行评估，该模式仅用于模型调试：

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --debug
```

输出打印示例：

```text
[2022-06-23 08:25:23,017] [    INFO] - -----------------------------
[2022-06-23 08:25:23,017] [    INFO] - Class name: 时间
[2022-06-23 08:25:23,018] [    INFO] - Evaluation precision: 1.00000 | recall: 1.00000 | F1: 1.00000
[2022-06-23 08:25:23,145] [    INFO] - -----------------------------
[2022-06-23 08:25:23,146] [    INFO] - Class name: 目的地
[2022-06-23 08:25:23,146] [    INFO] - Evaluation precision: 0.64286 | recall: 0.90000 | F1: 0.75000
[2022-06-23 08:25:23,272] [    INFO] - -----------------------------
[2022-06-23 08:25:23,273] [    INFO] - Class name: 费用
[2022-06-23 08:25:23,273] [    INFO] - Evaluation precision: 0.11111 | recall: 0.10000 | F1: 0.10526
[2022-06-23 08:25:23,399] [    INFO] - -----------------------------
[2022-06-23 08:25:23,399] [    INFO] - Class name: 出发地
[2022-06-23 08:25:23,400] [    INFO] - Evaluation precision: 1.00000 | recall: 1.00000 | F1: 1.00000
```

可配置参数说明：

- `model_path`: 进行评估的模型文件夹路径，路径下需包含模型权重文件`model_state.pdparams`及配置文件`model_config.json`。
- `test_path`: 进行评估的测试集文件。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `model`: 选择所使用的模型，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`和`uie-nano`，默认为`uie-base`。
- `debug`: 是否开启debug模式对每个正例类别分别进行评估，该模式仅用于模型调试，默认关闭。

#### 定制模型一键预测

`paddlenlp.Taskflow`装载定制模型，通过`task_path`指定模型权重文件的路径，路径下需要包含训练好的模型权重文件`model_state.pdparams`。

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow

>>> schema = ['出发地', '目的地', '费用', '时间']
# 设定抽取目标和定制化模型权重路径
>>> my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best')
>>> pprint(my_ie("城市内交通费7月5日金额114广州至佛山"))
[{'出发地': [{'end': 17,
           'probability': 0.9975287467835301,
           'start': 15,
           'text': '广州'}],
  '时间': [{'end': 10,
          'probability': 0.9999476678061399,
          'start': 6,
          'text': '7月5日'}],
  '目的地': [{'end': 20,
           'probability': 0.9998511131226735,
           'start': 18,
           'text': '佛山'}],
  '费用': [{'end': 15,
          'probability': 0.9994474579292856,
          'start': 12,
          'text': '114'}]}]
```

#### Few-Shot实验

我们在互联网、医疗、金融三大垂类自建测试集上进行了实验：

<table>
<tr><th row_span='2'><th colspan='2'>金融<th colspan='2'>医疗<th colspan='2'>互联网
<tr><td><th>0-shot<th>5-shot<th>0-shot<th>5-shot<th>0-shot<th>5-shot
<tr><td>uie-base (12L768H)<td><b>46.43</b><td><b>70.92</b><td><b>71.83</b><td><b>85.72</b><td><b>78.33</b><td><b>81.86</b>
<tr><td>uie-medium (6L768H)<td>41.11<td>64.53<td>65.40<td>75.72<td>78.32<td>79.68
<tr><td>uie-mini (6L384H)<td>37.04<td>64.65<td>60.50<td>78.36<td>72.09<td>76.38
<tr><td>uie-micro (4L384H)<td>37.53<td>62.11<td>57.04<td>75.92<td>66.00<td>70.22
<tr><td>uie-nano (4L312H)<td>38.94<td>66.83<td>48.29<td>76.74<td>62.86<td>72.35
</table>

0-shot表示无训练数据直接通过```paddlenlp.Taskflow```进行预测，5-shot表示基于5条标注数据进行模型微调。**实验表明UIE在垂类场景可以通过少量数据（few-shot）进一步提升效果**。

#### Python部署

以下是UIE Python端的部署流程，包括环境准备、模型导出和使用示例。

- 环境准备
  UIE的部署分为CPU和GPU两种情况，请根据你的部署环境安装对应的依赖。

  - CPU端

    CPU端的部署请使用如下命令安装所需依赖

    ```shell
    pip install -r deploy/python/requirements_cpu.txt
    ```

  - GPU端

    为了在GPU上获得最佳的推理性能和稳定性，请先确保机器已正确安装NVIDIA相关驱动和基础软件，确保**CUDA >= 11.2，cuDNN >= 8.1.1**，并使用以下命令安装所需依赖

    ```shell
    pip install -r deploy/python/requirements_gpu.txt
    ```

    如需使用半精度（FP16）部署，请确保GPU设备的CUDA计算能力 (CUDA Compute Capability) 大于7.0，典型的设备包括V100、T4、A10、A100、GTX 20系列和30系列显卡等。
    更多关于CUDA Compute Capability和精度支持情况请参考NVIDIA文档：[GPU硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)


- 模型导出

  将训练后的动态图参数导出为静态图参数：

  ```shell
  python export_model.py --model_path ./checkpoint/model_best --output_path ./export
  ```

  可配置参数说明：

  - `model_path`: 动态图训练保存的参数路径，路径下包含模型参数文件`model_state.pdparams`和模型配置文件`model_config.json`。
  - `output_path`: 静态图参数导出路径，默认导出路径为`./export`。

- 推理

  - CPU端推理样例

    在CPU端，请使用如下命令进行部署

    ```shell
    python deploy/python/infer_cpu.py --model_path_prefix export/inference
    ```

    可配置参数说明：

    - `model_path_prefix`: 用于推理的Paddle模型文件路径，需加上文件前缀名称。例如模型文件路径为`./export/inference.pdiparams`，则传入`./export/inference`。
    - `position_prob`：模型对于span的起始位置/终止位置的结果概率0~1之间，返回结果去掉小于这个阈值的结果，默认为0.5，span的最终概率输出为起始位置概率和终止位置概率的乘积。
    - `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
    - `batch_size`: 批处理大小，请结合机器情况进行调整，默认为4。

  - GPU端推理样例

    在GPU端，请使用如下命令进行部署

    ```shell
    python deploy/python/infer_gpu.py --model_path_prefix export/inference --use_fp16
    ```

    可配置参数说明：

    - `model_path_prefix`: 用于推理的Paddle模型文件路径，需加上文件前缀名称。例如模型文件路径为`./export/inference.pdiparams`，则传入`./export/inference`。
    - `use_fp16`: 是否使用FP16进行加速，默认关闭。
    - `position_prob`：模型对于span的起始位置/终止位置的结果概率0~1之间，返回结果去掉小于这个阈值的结果，默认为0.5，span的最终概率输出为起始位置概率和终止位置概率的乘积。
    - `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
    - `batch_size`: 批处理大小，请结合机器情况进行调整，默认为4。

<a name="CCKS比赛"></a>

## 5.CCKS比赛

为了进一步探索通用信息抽取的边界，我们举办了**CCKS 2022 千言通用信息抽取竞赛评测**（2022/03/30 - 2022/07/31）。

- [报名链接](https://aistudio.baidu.com/aistudio/competition/detail/161/0/introduction)
- [基线代码](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/DuUIE)

## References
- **[Unified Structure Generation for Universal Information Extraction](https://arxiv.org/pdf/2203.12277.pdf)**