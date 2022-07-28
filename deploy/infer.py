# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pprint import pprint
from uie_predictor import UIEPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--onnx_model_path",
        type=str,
        required=True,
        help="The path prefix of inference model to be used.",
    )
    parser.add_argument(
        "--position_prob",
        default=0.5,
        type=float,
        help="Probability threshold for start/end index probabiliry.",
    )
    parser.add_argument(
        "--max_seq_len",
        default=512,
        type=int,
        help="The maximum input sequence length. Sequences longer than this will be split automatically.",
    )
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="Batch size per CPU for inference.")
    parser.add_argument(
        "--device",
        default='cpu',
        type=str,
        help="The device to use.",
    )
    parser.add_argument(
        "--use_fp16",
        action='store_true',
        help=
        "Whether to use fp16 inference, only takes effect when deploying on gpu.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    texts = [
        '"为外地返京人员，常住地址为望京西园一区112号楼，集中隔离前居住地为朝阳区东坝乡华瀚福园C区409号楼。3月8日晚作为外省确诊病例密切接触者进行集中隔离，3月11日被判定为无症状感染者。    主要活动轨迹：    3月5日早上7:20，从上海虹桥机场乘坐航班MU5151，于10:35分到达首都机场T2航站楼，后乘私家车约中午12:00到顺义沃尔玛山姆会员店购物。16:30乘私家车回到东湖街道望京西园一区112楼家中。    3月6日约11:00，步行前往“北京国宗济世中医院”采样点进行核酸检测，随后步行返回家中；中午12:30左右，自行驾车前往东坝华瀚福园C区，居家未外出。当日核酸检测结果为阴性。    3月7日约10：30，自行驾车前往望京西园一区家中，11:00在单元门外取其行李后离开，约12:00回到东坝华瀚福园家中，后一直居家未外出。    3月8日，居家未外出，当晚作为外省一确诊病例的密切接触者进行集中隔离。    3月11日确诊为无症状感染者。"'
    ]

    schema = ['日期', '时间', '地点']

    args.schema = schema
    predictor = UIEPredictor(args)

    print("-----------------------------")
    outputs = predictor.predict(texts)
    for text, output in zip(texts, outputs):
        print("1. Input text: ")
        print(text)
        print("2. Input schema: ")
        print(schema)
        print("3. Result: ")
        pprint(output)
        print("-----------------------------")


if __name__ == "__main__":
    main()
