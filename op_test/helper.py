import os
import glob
import copy
import time
import onnxruntime
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from tools.profiler import get_model_capacity


class TestHelper(object):
    def __init__(self, model, dump_path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_name = model.__class__.__name__
        self.dump_path = dump_path

        self.input_names, self.output_names, self.onnx_file = self.export_to_onnx()

    def export_to_onnx(self):
        os.makedirs(self.dump_path, exist_ok=True)
        onnx_file = os.path.join(self.dump_path, f"{self.model_name}.onnx")

        inputs = self.model.get_random_inputs()
        input_names = [f"input{i}" for i in range(len(inputs))]
        output_names = [f"output{i}" for i in range(self.model.get_output_number())]

        torch.onnx.export(
            self.model,
            inputs,
            onnx_file,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            opset_version=16,
        )
        return input_names, output_names, onnx_file

    def run_onnx_inference(self, inputs):
        session = onnxruntime.InferenceSession(self.onnx_file)

        outputs = session.run(
            None,
            {
                name: data.cpu().detach().numpy().astype(np.float32)
                for name, data in zip(self.input_names, inputs)
            },
        )
        return outputs

    def write_input_data(self, data, npy_file):
        np.save(npy_file, data)

    def write_output_data(self, data, txt_file):
        with open(txt_file, "w") as fp:
            for x in data.flatten():
                fp.write(f"{float(x)}\n")

    def glob_input_files(self):
        n_inputs = len(self.model.get_random_inputs())

        for j in range(n_inputs):
            txt_file = os.path.join(self.dump_path, f"dataset{j}.txt")

            input_files = sorted(
                glob.glob(os.path.join(self.dump_path, "data", f"*_input{j}.npy")),
                key=lambda x: int(os.path.basename(x).split("_")[0]),
            )

            with open(txt_file, "w") as fp:
                for input_file in input_files:
                    fp.write(os.path.relpath(input_file, self.dump_path) + "\n")

    def execute(self, n=20, *args, **kwargs):
        data_path = os.path.join(self.dump_path, "data")
        os.makedirs(data_path, exist_ok=True)

        fps = []
        for i in tqdm(range(n), desc=f"random test MODEL={self.model_name}..."):
            inputs = self.model.get_random_inputs()
            inputs = [x.to(self.device) for x in inputs]

            if i == 0:
                _ = get_model_capacity(self.model, tuple(inputs), verbose=True)

            for j, data in enumerate(inputs):
                self.write_input_data(
                    data.cpu().detach().numpy(),
                    os.path.join(data_path, f"{i:04d}_input{j}.npy"),
                )

            start = time.time()
            torch_outputs = self.model(*inputs)
            end = time.time()
            fps.append(float(end - start))

            torch_outputs = [y.detach().cpu().numpy() for y in torch_outputs]
            onnx_outputs = self.run_onnx_inference(inputs)

            for k, (x, y) in enumerate(zip(torch_outputs, onnx_outputs)):
                assert np.allclose(
                    x,
                    y,
                    rtol=1e-6,
                    atol=1e-6,
                ), f"ITER-{i} OUTPUT-{k} torch and onnx mismatch!"

            for k, data in enumerate(torch_outputs):
                self.write_output_data(
                    data,
                    os.path.join(data_path, f"{i:04d}_output{k}.txt"),
                )
        print(
            f"Device={self.device}, OP={self.model.__class__.__name__}, Median FPS = {(1.0 / np.median(fps)):.4f} for {n} inferences."
        )

        self.glob_input_files()
