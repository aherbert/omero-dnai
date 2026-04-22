#!/usr/bin/env python3
"""Analyse OMERO images using DNAi."""

import argparse

from dnafiber.model.models_zoo import MODELS_ZOO


def main():
    parser = argparse.ArgumentParser(
        description="Analyse OMERO images using DNAi."
    )
    _ = parser.add_argument(
        "id",
        type=int,
        nargs="+",
        help="Image/Dataset ID",
    )
    _ = parser.add_argument(
        "--c1",
        type=int,
        default=1,
        help="Channel 1 (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--c2",
        type=int,
        default=0,
        help="Channel 2 (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--model",
        choices=list(MODELS_ZOO.keys()),
        help="Model (default to ensemble)",
    )
    _ = parser.add_argument(
        "--tta",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use test time augmentation (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--device",
        help="Torch device (default to auto-detect)",
    )
    _ = parser.add_argument(
        "--results",
        help="Result filename (default to use input IDs)",
    )
    _ = parser.add_argument(
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing results (default: %(default)s)",
    )
    args = parser.parse_args()

    import os

    fn = args.results
    if fn is None:
        if len(args.id) == 1:
            fn = f"results_{args.id[0]}.csv"
        else:
            fn = f"results_{args.id[0]}_and_{len(args.id) - 1}.csv"
    if os.path.exists(fn):
        if args.overwrite:
            print(f"Warning: results file {fn} already exists, will be overwritten")
        else:
            print(f"Warning: results file {fn} already exists")
            exit(1)
    print(f"Results file: {fn}")

    import numpy as np
    import torch
    import pandas as pd
    import time
    from omero.cli import cli_login
    from omero.gateway import BlitzGateway

    # Workaround for streamlit logger warnings:
    # https://discuss.streamlit.io/t/warning-for-missing-scriptruncontext/83893/16
    import streamlit
    import logging
    for name, l in logging.root.manager.loggerDict.items():
        if "streamlit" in name:
            l.disabled = True

    # Run DNAi on the image
    from dnafiber.data.readers import format_raw_image
    from dnafiber.deployment import run_one_file
    from dnafiber.model.utils import get_ensemble_models, get_error_detection_model, _get_model
    from dnafiber.data.preprocess import preprocess

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"Using device={device}")

    # Load model
    model = MODELS_ZOO.get(args.model)
    if model is None:
        model = get_ensemble_models()
        print(f"Using model=ensemble")
    else:
        model = _get_model(model)
        print(f"Using model={args.model}")
    error_model = get_error_detection_model()

    # This must be sent to the device for the error detection step
    if isinstance(model, list):
        for m in model:
            m.to(device)
    else:
        model = model.to(device)
    error_model = error_model.to(device)

    def get_image_ids(conn, id):
        dataset = conn.getObject("Dataset", id)
        if dataset is not None:
            return [img.getId() for img in dataset.listChildren()]
        image = conn.getObject("Image", id)
        if image is None:
            print(f"Warning: Image {id} not found")
            return []
        return [image.getId()]

    dfs = []

    print(f"Connecting to OMERO")
    with cli_login() as cli:
        start = time.time()
        conn = BlitzGateway(client_obj=cli._client)
        # Set group to -1 for all groups
        conn.SERVICE_OPTS.setOmeroGroup(-1)

        # Process all images
        for input_id in args.id:
            for image_id in get_image_ids(conn, input_id):
                image = conn.getObject("Image", image_id)

                sizeZ = image.getSizeZ()
                sizeC = image.getSizeC()
                sizeT = image.getSizeT()
                print(f"Image {image_id}: z={sizeZ}, c={sizeC}, t={sizeT}")

                if sizeZ * sizeT != 1 or sizeC == 1:
                    print(f"Warning: Require 2D image with channels")
                    continue

                pp = image.getPrimaryPixels()
                img = np.array([pp.getPlane(0, args.c1, 0), pp.getPlane(0, args.c2, 0)])

                pixel_size = image.getPixelSizeX("MICROMETER").getValue()

                image_name = image.getName()
                print(f"Image {image_id} [{image_name}]: shape={img.shape}, pixel_size={pixel_size}")

                img = format_raw_image(img)
                img = preprocess(img, pixel_size=pixel_size)

                # Run inference on a single image
                fibers = run_one_file(
                    # Convert to uint8 and make YXC with 3 channels
                    img,
                    model=model,
                    pixel_size=pixel_size,  # µm/pixel
                    use_tta=args.tta,
                    error_detection_model=error_model,
                    device=device,
                )

                # Filter and export
                fibers = fibers.valid_copy()  # Keep only classifiable fibers
                fibers = fibers.filter_errors(threshold=args.threshold)  # Remove likely false positives

                df = fibers.to_df(pixel_size=pixel_size, img_name=image_name)
                df["Image ID"] = image_id
                dfs.append(df)
        print(f"Image processing time: {time.time() - start:.2f} seconds")

    if dfs:
        print(f"Saving results file: {fn}")
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(fn, index=False)
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
