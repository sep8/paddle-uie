import os
import argparse
import paddle2onnx

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, default='./model', help="The path to model parameters to be loaded.")
parser.add_argument("--output_path", type=str, default='./model', help="The path of onnx model to be saved.")
parser.add_argument( "--use_fp16", default=False, type=bool, help="Whether use fp16.")

args = parser.parse_args()

if __name__ == "__main__":
  model_path = args.model_path
  use_fp16 = args.use_fp16

  model_file = os.path.join(model_path, "inference.pdmodel")
  params_file = os.path.join(model_path, "inference.pdiparams")

  output_path = args.output_path
  onnx_file = os.path.join(output_path, "model.onnx")

  onnx_model = paddle2onnx.command.c_paddle_to_onnx(
            model_file=model_file,
            params_file=params_file,
            opset_version=13,
            enable_onnx_checker=True)
  with open(onnx_file, "wb") as f:
      f.write(onnx_model)

  if use_fp16:
    from onnxconverter_common import float16
    import onnx
    fp16_model_file = os.path.join(output_path, "fp16_model.onnx")
    onnx_model = onnx.load_model(onnx_file)
    trans_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
    onnx.save_model(trans_model, fp16_model_file)
    onnx_model = fp16_model_file
