from flask import Flask, request, jsonify
import argparse
from uie_predictor import UIEPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--onnx_model_path",
        default="./static",
        type=str,
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
    args, unknown = parser.parse_known_args()
    return args

# Change to the schema you want
schema = ['感染者', '日期', '时间', '地点']

app = Flask(__name__, static_folder='static')


@app.route("/", methods=['POST'])
def ner():
    text = request.form.get("text") or request.json.get("text") or request.values.get("text")
    args = parse_args()
    args.schema = schema
    predictor = UIEPredictor(args)
    print(F">>> [InferBackend] Use {args.device} to inference ...")
    outputs = predictor.predict([text])
    print("-----------------------------------------------------")
    entities = []
    for entity in outputs:
        for k, v in entity.items():
            for item in v:
                entities.append({
                    'label': k,
                    'start': item['start'],
                    'end': item['end'],
                    'text': item['text']
                })
    return jsonify(entities)
