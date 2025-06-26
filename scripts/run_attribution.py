import argparse
import copy
import glob
import json
from typing import Any, Dict, List, Union

import numpy as np
import torch
from form_source_attribution.src.model.automodel import automodel
from PIL import Image


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Run attribution tasks with a model on a specific dataset."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model to use for attribution"
    )
    parser.add_argument(
        "--llm_inference",
        type=str,
        required=True,
        help="Input llm inferences in a dictionary using the document names as keys. The inference is a dictionary.",
    )
    parser.add_argument(
        "--ocr",
        type=str,
        required=True,
        help="Input ocr inference.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for results"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="The path containing all the images required for inference",
    )
    parser.add_argument(
        "--patches", type=int, required=True, help="The number of patches per image"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        required=True,
        help="The percentage overlap between patches",
    )
    return parser.parse_args()


def extract_patches(image: Image.Image, num_patches: int = 3, overlap: float = 0.3):
    """
    Divide an image into patches with specified overlap.

    Parameters:
    - image: PIL.Image.Image object
    - num_patches: Number of patches to create (default is 3)
    - overlap: Fractional overlap between patches (0.0 to <1.0)

    Returns:
    - List of PIL.Image.Image patches
    """
    if not (0 <= overlap < 1):
        raise ValueError("Overlap must be between 0 (inclusive) and 1 (exclusive).")

    width, height = image.size
    img_array = np.array(image)
    if img_array.ndim == 2:  # Grayscale image
        img_array = img_array[:, :, np.newaxis]  # Add channel dim for consistency

    all_patches = []
    all_coords = []

    patch_dim = height / (1 + (num_patches - 1) * (1 - overlap))  # The patch dimension
    step = patch_dim * (
        1 - overlap
    )  # The step, i.e. how much to move to initiate the next patch

    for i in range(num_patches):
        x0 = max(0, int(i * step))
        x1 = min(height, int(x0 + patch_dim))
        patch_array = img_array[x0:x1, 0:width]
        patch = (
            Image.fromarray(patch_array.squeeze())
            if patch_array.shape[2] == 1
            else Image.fromarray(patch_array)
        )

        all_patches.append(patch)
        all_coords.append((0, x0, width, x1))

    return all_patches, all_coords


def filter_keyvalue_pairs(
    input_dict: Dict[str, Union[Dict, str]],
) -> Dict[str, Dict[Any, Any] | str]:
    final_dict = {}

    for key, item_dict in input_dict.items():
        if (item_dict["value"] is None) or (item_dict["value"] == "None"):
            continue
        if item_dict["value"].strip() == "":
            continue
        final_dict[key] = item_dict

    return final_dict


def create_query(
    key: str,
    new_key: str,
    value: str,
) -> str:
    if "]" in new_key:
        new_processed_key = new_key.split("]")[1]
    else:
        new_processed_key = new_key
    new_processed_key = new_processed_key.replace("_", " ").strip()

    if "_<R" in key:
        processed_key = key.split("_<R")[0]
    else:
        processed_key = key
    processed_key = processed_key.replace("_", " ").replace(".", " ").strip()

    query = f"[{processed_key}]{new_processed_key} {value}"

    return query


def run_inference(
    query: str,
    # actual_value: str,
    images: List[Image.Image],
    images_pages: List[int],
    matcher: Any,
    split_number: int = 3,
    overlap: float = 0.3,
):
    old_images = []
    for idx, img in zip(images_pages, images):
        old_images.append(
            {
                "image": img,
                "score": -1e6,
                "coordinates": [0, 0, img.width, img.height],
                "page_n": idx,
                "requires_processing": True,
            }
        )

    all_inferences = []
    for old_img in old_images:
        # try:
        inferences = single_img_inference(
            query, old_img, matcher, split_number, overlap
        )
        all_inferences.extend(inferences)
        old_img["requires_processing"] = False
    all_inferences = old_images + all_inferences
    all_inferences = sorted(
        all_inferences, key=lambda x: x["score"], reverse=True
    )  # [:top_k]
    target_inferences = []
    for line in all_inferences:
        if line not in target_inferences:
            target_inferences.append(line)
    all_inferences = target_inferences
    last_score = all_inferences[0]["score"]
    all_inferences = list(filter(lambda x: x["score"] >= last_score, all_inferences))

    return all_inferences


def single_img_inference(
    query: str,
    # modified_value: str,  # this is the actual value we are looking for
    img: Dict[str, Any],
    matcher: Any,
    split_number: int,
    overlap: float,
):
    if not img["requires_processing"]:  # Inference on image splits was already done
        return [img]

    new_image = copy.copy(img["image"])
    all_patches, all_coords = extract_patches(new_image, split_number, overlap)

    # Create image coordinates
    modified_coordinates = []
    for line in all_coords:
        modified_coordinates.append(
            [
                line[0] + img["coordinates"][0],
                line[1] + img["coordinates"][1],
                line[2] + img["coordinates"][0],
                line[3] + img["coordinates"][1],
            ]
        )
    all_coords = modified_coordinates

    # Create patches embeddings
    matcher.embed_patches(all_patches, all_coords, [img["page_n"]] * len(all_coords))

    values, indexes = matcher.find_best_match(
        query,
        targets=[[coord, img["page_n"]] for coord in all_coords],
        top_n=len(all_coords),
    )
    val_ids = [[value, idx] for value, idx in zip(values, indexes)]
    val_ids.sort(key=lambda x: x[1], reverse=False)

    output = []
    for (score, _), coord, patch in zip(val_ids, all_coords, all_patches):
        output.append(
            {
                "image": patch,
                "score": float(score),
                "coordinates": coord,
                "page_n": img["page_n"],
                "requires_processing": True,
            }
        )

    return output


def main():
    args = argument_parser()

    img_dir = args.images

    # Load the model
    embedder = automodel(
        model_name=args.model, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the dataset
    with open(args.llm_inference, "r") as f:
        llm_inference = json.load(f)

    # Loaf the OCR
    with open(args.ocr, "r") as f:
        dataset_ocr = json.load(f)

    predictions = []  # We store all predictions in here
    for doc_name, file_inference in llm_inference.items():
        keyvalue_pairs = filter_keyvalue_pairs(file_inference)

        images_list = glob.glob(f"{img_dir}/{doc_name}__*.jpg")
        images_list = sorted(
            images_list, key=lambda x: int(x.split("__")[-1].replace(".jpg", ""))
        )
        images = [Image.open(img_path) for img_path in images_list]

        for key, item_dict in keyvalue_pairs.items():
            value = item_dict["value"]
            new_key = item_dict["key"]

            try:
                page = int(item_dict["page"])
            except Exception:
                page = None

            query = create_query(key, new_key, value)

            if (page is not None) and (page < len(images)):
                filtered_images = [images[page - 1]]
                filtered_page_numbers = [page - 1]
            else:
                filtered_images = images
                filtered_page_numbers = list(range(len(images)))

            pages_ocr = []
            ocr_dict = dataset_ocr[doc_name]

            for page_idx in filtered_page_numbers:
                pages_ocr.append(ocr_dict[str(page_idx + 1)])
            # actual_value = value.replace(" ", "")

            inferences = run_inference(
                query,
                # actual_value,
                filtered_images,
                filtered_page_numbers,
                embedder,
                args.patches,
                args.overlap,
            )

            for line in inferences:
                predictions.append(
                    {
                        "label": key,
                        "value": value,
                        "bbox": line["coordinates"],
                        "page_number": line["page_n"],
                        "score": line["score"],
                        "doc_name": doc_name,
                        "image": line["image"],
                    }
                )

    with open(args.output, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
