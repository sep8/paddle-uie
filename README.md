# UIE(Universal Information Extraction) with [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)

#### 可用的模型

- 模型选择

  | 模型 |  结构  |
  | :---: | :--------: |
  | `uie-base` (默认)| 12-layers, 768-hidden, 12-heads |
  | `uie-medical-base` | 12-layers, 768-hidden, 12-heads |
  | `uie-medium`| 6-layers, 768-hidden, 12-heads |
  | `uie-mini`| 6-layers, 384-hidden, 12-heads |
  | `uie-micro`| 4-layers, 384-hidden, 12-heads |
  | `uie-nano`| 4-layers, 312-hidden, 12-heads |

#### 运行环境

- 可以直接使用C3-NLP的的python虚拟环境

<a name="轻定制功能"></a>

##  轻定制功能

对于细分场景我们推荐使用轻定制功能（标注少量数据进行模型微调）以进一步提升效果。下面通过`报销工单信息抽取`的例子展示如何通过5条训练数据进行UIE模型微调。

#### [代码结构](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie)

```shell
.
├── data              # 原始数据和转换后的训练数据
├────raw             
├─────raw.txt         # 用到导入到 `doccano` 进行标注的示例数据
├── utils.py          # 数据处理工具
├── model.py          # 模型组网脚本
├── doccano.py        # 数据标注脚本
├── doccano.md        # 数据标注文档
├── finetune.py       # 模型微调脚本
├── evaluate.py       # 模型评估脚本
├── deploy            # 模型部署相关脚本
├──── export_model.py   # 将训练后的动态图参数导出为静态图参数
├──── to_onnx.py        # 将静态图参数转换为onnx模型
├──── uie_predictor     # 调用onnx模型做推理
├──── infer.py          # 调用`uie_predictor`做推理（预定义了测试文本）
├──── app.py            # 展示如何使用 `flask` 将onnx模型部署到服务端
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

- 标注完成后，在doccano平台上导出文件，并将其重命名为``doccano_ext.jsonl``后，放入``./data/raw``目录下。

- 这里我们提供预先标注好的文件[doccano_ext.jsonl](https://bj.bcebos.com/paddlenlp/datasets/uie/doccano_ext.json)，可直接下载并放入`./data/raw`目录。执行以下脚本进行数据转换，执行后会在`./data`目录下生成训练/验证/测试集文件。

```shell
python doccano.py \
    --doccano_file ./data/raw/doccano_ext.jsonl \
    --task_type ext \
    --save_dir ./data \
    --splits 0.8 0.1 0.1
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
    --num_epochs 10 \
    --model uie-mini \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device cpu
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
#### Python部署

以下是UIE Python端的部署流程，包括环境准备、模型导出和使用示例。

- 环境准备
  UIE的部署分为CPU和GPU两种情况，请根据你的部署环境安装对应的依赖。

  - CPU端

    CPU端的部署请使用如下命令安装所需依赖

    ```shell
    pip install onnx onnxruntime
    ```

  - GPU端

    为了在GPU上获得最佳的推理性能和稳定性，请先确保机器已正确安装NVIDIA相关驱动和基础软件，确保**CUDA >= 11.2，cuDNN >= 8.1.1**，并使用以下命令安装所需依赖

    ```shell
    pip install onnx onnxconverter_common onnxruntime-gpu
    ```

    如需使用半精度（FP16）部署，请确保GPU设备的CUDA计算能力 (CUDA Compute Capability) 大于7.0，典型的设备包括V100、T4、A10、A100、GTX 20系列和30系列显卡等。
    更多关于CUDA Compute Capability和精度支持情况请参考NVIDIA文档：[GPU硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)


- 模型导出

  将训练后的动态图参数导出为静态图参数：

  ```shell
  python deploy/export_model.py --model_path ./checkpoint/model_best --output_path ./model
  ```

  可配置参数说明：

  - `model_path`: 动态图训练保存的参数路径，路径下包含模型参数文件`model_state.pdparams`和模型配置文件`model_config.json`。
  - `output_path`: 静态图参数导出路径，默认导出路径为`./model`。

- 模型转换

  将静态图参数模型转换为onnx模型：

  ```shell
  python deploy/to_onnx.py --model_path ./model --output_path ./model
  ```

  可配置参数说明：

  - `model_path`: 静态图参数路径，路径下包含模型参数文件`inference.pdiparams`和模型配置文件`inference.pdmodel`。
  - `output_path`: 静态图参数导出路径，默认导出路径为`./model`。


- 推理
  封装了一个用来做UIE推理的 CLASS `deploy/uie_predictor`, `infer.py`会调用 `uie_predictor`做一个简单的推理（使用的预定义好的`schema`和文本）

  - CPU端推理样例

    在CPU端，请使用如下命令进行部署

    ```shell
    python deploy/infer.py --onnx_model_path ./model
    ```

    在GPU端，请使用如下命令进行部署

    ```shell
    python deploy/infer.py --onnx_model_path ./model --device gpu
    ```

    可配置参数说明：

    - `onnx_model_path`: 用于推理的onnx模型文件路径，文件名定义为 `model.onnx`。例如模型文件路径为`./model/model.onnx`，则传入`./model`。
    - `position_prob`：模型对于span的起始位置/终止位置的结果概率0~1之间，返回结果去掉小于这个阈值的结果，默认为0.5，span的最终概率输出为起始位置概率和终止位置概率的乘积。
    - `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
    - `batch_size`: 批处理大小，请结合机器情况进行调整，默认为4。

- 云部署
  训练后的模型参数比较大，无法直接提交到github上，所以可以选择将转换后的onnx模型放到云端
  - 将onnx模型放到 ``[google drive](https://drive.google.com/drive/)``
  - 将访问权限修改为 ``Anyone with the link``
  - 下载模型
    - 安装gdown ``pip install gdown``
    ```python
    import gdown

    id = 'The file id for shared "model.onnx"'
    output = '<path>/to/model.onnx'
    proxy = 'Your proxy url' # http://proxy.****.com:****
    gdown.download(id=id, output=output, quiet=False, proxy=proxy)
    ```
- 训练好的模型
  - Google drive 上放置了一个训练好的 [onnx模型](https://drive.google.com/file/d/1VnvD72MbF678fFxw78dBGz6MIu7i-Elr/view?usp=sharing)
  - schema
    - 实体
      - 日期，时间，地点（住址， 单位），指代（家，学校等）
    - 关系
      - 出行时间
      - 出行地点
  - 特殊处理：
    - “感染者”和轨迹关系，需要通过数据预处理确定
    - 只有“日期”没有“时间”的“地点”，不标注关系
    - 忽略不明确的“日期”“时间”实体，比如 近两周，每日
    - 上午，中午，下午，晚上 等词语标注为“时间”
    - “xx月xx日至xx日“，或 “xx月xx日，xx日“，整体标记为时间，在模型输出结果进行进一步处理 

- 使用训练好的模型做 `doccano` 的 `Auto labeling`
  - 安装flask `pip install flask`
  - 将 `model/model.onnx` 移动到 `deploy/static`
  - 打开文件 `deploy/app.py` ，修改`schema`
  - 运行`flask`, 进入 `deploy`文件夹，然后运行
  ```shell
    flask run
  ```
  - 启动`doccano`，然后配置`Auto labeling`，可以参考[官方文档](https://doccano.github.io/doccano/advanced/auto_labelling_config/), [How to connect to a local REST API for auto annotation](https://github.com/doccano/doccano/issues/1417)

## References
- **[Unified Structure Generation for Universal Information Extraction](https://arxiv.org/pdf/2203.12277.pdf)**
