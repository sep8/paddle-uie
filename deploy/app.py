from flask import Flask, request, jsonify
import argparse
from uie_predictor import UIEPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_path_prefix",
        default="static/export/inference",
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
    args, unknown = parser.parse_known_args()
    return args


schema = ['感染者', '日期', '时间', '地点']

app = Flask(__name__, static_folder='static')


@app.route("/", methods=['POST'])
def ner():
    text = request.form.get("text") or request.json.get(
        "text") or request.values.get("text")
    print('-----------------------')
    args = parse_args()
    args.device = 'cpu'
    args.schema = schema
    predictor = UIEPredictor(args)
    outputs = predictor.predict([text])
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
