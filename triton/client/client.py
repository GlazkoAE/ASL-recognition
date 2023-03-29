import argparse
import sys

import numpy as np
import tritonclient.grpc as tritongrpcclient

from config import AppConfig


def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.
    """
    return np.fromfile(img_path, dtype="uint8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=False, default="ensemble", help="Model name"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    args = parser.parse_args()

    config: AppConfig = AppConfig.parse_raw()
    url = config.server_host + ":" + str(config.grpc_endpoint)

    try:
        triton_client = tritongrpcclient.InferenceServerClient(
            url=url, verbose=args.verbose
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    with open("labels.txt") as f:
        labels_dict = {idx: line.strip() for idx, line in enumerate(f)}

    inputs = []
    outputs = []
    input_name = "INPUT"
    output_name = "OUTPUT"
    image_data = load_image(args.image)
    image_data = np.expand_dims(image_data, axis=0)

    inputs.append(tritongrpcclient.InferInput(input_name, image_data.shape, "UINT8"))
    outputs.append(tritongrpcclient.InferRequestedOutput(output_name))

    inputs[0].set_data_from_numpy(image_data)
    results = triton_client.infer(
        model_name=args.model_name, inputs=inputs, outputs=outputs
    )

    output0_data = results.as_numpy(output_name)
    maxs = np.argmax(output0_data, axis=1)
    print("Result is class: {}".format(labels_dict[maxs[0]]))
