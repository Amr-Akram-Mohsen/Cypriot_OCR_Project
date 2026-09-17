"""
generate_dataset.py
===================

Cypriot Syllabary Synthetic Dataset Generator
for YOLO symbol-detection pre-training.

Generates:
    dataset/
        yolo_synthetic/
            images/
                train/
                val/
                test/
            labels/
                train/
                val/
                test/

Each image contains multiple Cypriot syllabary symbols.

Each corresponding .txt file contains YOLO annotations:

    class_id center_x center_y width height

All coordinates are normalized to [0, 1].

The synthetic images are intended for:
    1. YOLO pre-training on synthetic data
    2. Later YOLO fine-tuning on manually annotated real images
"""

import os
import random
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 1. CONFIGURATION
# ============================================================

FONT_PATH = os.path.join(
    "assets",
    "fonts",
    "NotoSansCypriot-Regular.ttf"
)

OUTPUT_DIR = os.path.join(
    "dataset",
    "yolo_synthetic"
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

# Number of full synthetic images in each split.
TRAIN_IMAGES = 3000
VAL_IMAGES = 600
TEST_IMAGES = 300

# Number of symbols placed into each image.
MIN_SYMBOLS_PER_IMAGE = 12
MAX_SYMBOLS_PER_IMAGE = 35

# Internal rendering resolution for individual glyphs.
GLYPH_RENDER_SIZE = 256

# Font size range used before scaling glyphs.
MIN_FONT_SIZE = 150
MAX_FONT_SIZE = 220

OVERWRITE = True

# Probability of each degradation level.
TIER_PROBS = {
    "clean": 0.25,
    "moderate": 0.55,
    "heavy": 0.20,
}

# Minimum distance between glyph bounding boxes.
MIN_BOX_GAP = 5

# Maximum number of placement attempts for one glyph.
MAX_PLACEMENT_ATTEMPTS = 100


# ============================================================
# 2. CLASS REPERTOIRE
# ============================================================

CLASS_NAMES = [
    "a",
    "e",
    "i",
    "ja",
    "jo",
    "ka",
    "ke",
    "ki",
    "ko",
    "ku",
    "la",
    "le",
    "li",
    "lo",
    "lu",
    "ma",
    "me",
    "mi",
    "mo",
    "mu",
    "na",
    "ne",
    "ni",
    "no",
    "nu",
    "o",
    "pa",
    "pe",
    "pi",
    "po",
    "pu",
    "ra",
    "re",
    "ri",
    "ro",
    "ru",
    "sa",
    "se",
    "si",
    "so",
    "su",
    "ta",
    "te",
    "ti",
    "to",
    "tu",
    "u",
    "wa",
    "we",
    "wi",
    "wo",
    "xa",
    "xe",
    "za",
    "zo",
]


# ============================================================
# 3. DISCOVER UNICODE CHARACTERS
# ============================================================

def get_cypriot_characters():
    """
    Find Unicode characters corresponding to CLASS_NAMES.

    Returns:
        dict:
            label -> Unicode character
    """

    discovered = {}

    for codepoint in range(0x10800, 0x10840):
        char = chr(codepoint)

        try:
            name = unicodedata.name(char)
        except ValueError:
            continue

        if name.startswith("CYPRIOT SYLLABLE "):
            label = name.replace(
                "CYPRIOT SYLLABLE ",
                ""
            ).lower()

            discovered[label] = char

    missing = [
        name
        for name in CLASS_NAMES
        if name not in discovered
    ]

    if missing:
        raise RuntimeError(
            "Could not find Unicode characters for: "
            + ", ".join(missing)
        )

    return {
        name: discovered[name]
        for name in CLASS_NAMES
    }


# ============================================================
# 4. RANDOM DEGRADATION TIER
# ============================================================

def choose_tier():
    r = random.random()
    cumulative = 0.0

    for tier, probability in TIER_PROBS.items():
        cumulative += probability

        if r < cumulative:
            return tier

    return "moderate"


# ============================================================
# 5. GLYPH MASK
# ============================================================

def render_glyph_mask(char, tier):
    """
    Render one actual Unicode Cypriot glyph.

    Returns:
        mask
        bbox of the non-zero glyph
    """

    font_size = random.randint(
        MIN_FONT_SIZE,
        MAX_FONT_SIZE
    )

    font = ImageFont.truetype(
        FONT_PATH,
        font_size
    )

    canvas = Image.new(
        "L",
        (GLYPH_RENDER_SIZE, GLYPH_RENDER_SIZE),
        0
    )

    draw = ImageDraw.Draw(canvas)

    bbox = draw.textbbox(
        (0, 0),
        char,
        font=font
    )

    glyph_w = bbox[2] - bbox[0]
    glyph_h = bbox[3] - bbox[1]

    if glyph_w <= 0 or glyph_h <= 0:
        return None

    x = (
        GLYPH_RENDER_SIZE - glyph_w
    ) // 2 - bbox[0]

    y = (
        GLYPH_RENDER_SIZE - glyph_h
    ) // 2 - bbox[1]

    draw.text(
        (x, y),
        char,
        font=font,
        fill=255
    )

    mask = np.array(canvas)

    ys, xs = np.where(mask > 10)

    if len(xs) == 0:
        return None

    cropped = mask[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    return cropped


# ============================================================
# 6. GLYPH TRANSFORMATIONS
# ============================================================

def transform_glyph(mask, tier):
    """
    Apply controlled geometric and visual variation
    without changing the identity of the glyph.
    """

    h, w = mask.shape

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    target_size = random.randint(38, 105)

    scale = target_size / max(h, w)

    new_w = max(
        8,
        int(w * scale)
    )

    new_h = max(
        8,
        int(h * scale)
    )

    mask = cv2.resize(
        mask,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Aspect-ratio variation
    # --------------------------------------------------------

    sx = random.uniform(0.90, 1.10)
    sy = random.uniform(0.90, 1.10)

    new_w = max(
        8,
        int(mask.shape[1] * sx)
    )

    new_h = max(
        8,
        int(mask.shape[0] * sy)
    )

    mask = cv2.resize(
        mask,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # Slight rotation
    # --------------------------------------------------------

    max_angle = {
        "clean": 2.5,
        "moderate": 5.0,
        "heavy": 8.0,
    }[tier]

    angle = random.uniform(
        -max_angle,
        max_angle
    )

    center = (
        mask.shape[1] / 2,
        mask.shape[0] / 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    mask = cv2.warpAffine(
        mask,
        matrix,
        (
            mask.shape[1],
            mask.shape[0]
        ),
        flags=cv2.INTER_LINEAR,
        borderValue=0
    )

    # --------------------------------------------------------
    # Stroke-weight variation
    # --------------------------------------------------------

    weight_probability = random.random()

    if weight_probability < 0.20:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2, 2)
        )

        eroded = cv2.erode(
            mask,
            kernel,
            iterations=1
        )

        if np.count_nonzero(eroded) > 0:
            mask = eroded

    elif weight_probability > 0.80:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2, 2)
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=1
        )

    # --------------------------------------------------------
    # Mild perspective distortion
    # --------------------------------------------------------

    if random.random() < 0.35:

        h, w = mask.shape

        amount = random.uniform(
            0.02,
            0.07
        )

        source = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ])

        destination = np.float32([
            [
                random.uniform(-amount, amount) * w,
                random.uniform(-amount, amount) * h,
            ],
            [
                w - 1 + random.uniform(-amount, amount) * w,
                random.uniform(-amount, amount) * h,
            ],
            [
                w - 1 + random.uniform(-amount, amount) * w,
                h - 1 + random.uniform(-amount, amount) * h,
            ],
            [
                random.uniform(-amount, amount) * w,
                h - 1 + random.uniform(-amount, amount) * h,
            ],
        ])

        matrix = cv2.getPerspectiveTransform(
            source,
            destination
        )

        mask = cv2.warpPerspective(
            mask,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderValue=0
        )

    # --------------------------------------------------------
    # Degradation
    # --------------------------------------------------------

    if tier in ("moderate", "heavy"):

        dropout_probability = {
            "moderate": 0.25,
            "heavy": 0.45,
        }[tier]

        if random.random() < dropout_probability:

            dropout_rate = {
                "moderate": random.uniform(0.01, 0.04),
                "heavy": random.uniform(0.03, 0.08),
            }[tier]

            noise = (
                np.random.rand(
                    mask.shape[0],
                    mask.shape[1]
                ) < dropout_rate
            )

            mask = mask.copy()
            mask[noise] = 0

    return mask


# ============================================================
# 7. STONE BACKGROUND
# ============================================================

def create_stone_background(tier):
    """
    Create a grayscale stone-like background.
    """

    base = random.uniform(
        110,
        170
    )

    bg = np.full(
        (
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        ),
        base,
        dtype=np.float32
    )

    coarse_std, mid_std, fine_std = {
        "clean": (3.0, 2.0, 2.5),
        "moderate": (5.0, 3.5, 4.0),
        "heavy": (7.0, 5.0, 6.0),
    }[tier]

    # Coarse texture
    coarse = np.random.normal(
        0,
        coarse_std,
        (20, 20)
    ).astype(np.float32)

    bg += cv2.resize(
        coarse,
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        ),
        interpolation=cv2.INTER_CUBIC
    )

    # Medium texture
    medium = np.random.normal(
        0,
        mid_std,
        (80, 80)
    ).astype(np.float32)

    bg += cv2.resize(
        medium,
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        ),
        interpolation=cv2.INTER_LINEAR
    )

    # Fine texture
    bg += np.random.normal(
        0,
        fine_std,
        (
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        )
    ).astype(np.float32)

    # Large illumination gradient
    x_gradient = np.linspace(
        random.uniform(-10, 5),
        random.uniform(-5, 10),
        IMAGE_WIDTH
    )

    y_gradient = np.linspace(
        random.uniform(-10, 5),
        random.uniform(-5, 10),
        IMAGE_HEIGHT
    )

    bg += x_gradient[None, :]
    bg += y_gradient[:, None]

    # Occasional stone lines / imperfections
    if random.random() < 0.35:

        for _ in range(
            random.randint(1, 3)
        ):

            y = random.randint(
                0,
                IMAGE_HEIGHT - 1
            )

            cv2.line(
                bg,
                (0, y),
                (
                    IMAGE_WIDTH - 1,
                    y + random.randint(-5, 5)
                ),
                random.uniform(-15, 8),
                random.randint(1, 3)
            )

    return np.clip(
        bg,
        50,
        220
    ).astype(np.float32)


# ============================================================
# 8. PLACE GLYPH AS A CARVED RELIEF
# ============================================================

def carve_glyph(
    image,
    mask,
    x,
    y,
    tier
):
    """
    Apply a simple carved/grooved appearance.

    Returns:
        modified image
    """

    h, w = mask.shape

    region = image[
        y:y + h,
        x:x + w
    ].astype(np.float32)

    normalized = (
        mask.astype(np.float32) / 255.0
    )

    # Slightly irregular groove depth.
    local_noise = np.random.normal(
        0,
        random.uniform(0.8, 2.5),
        (h, w)
    ).astype(np.float32)

    depth = normalized * (
        random.uniform(35, 65)
        + local_noise
    )

    carved = region - depth

    # Directional relief.
    grad_x = cv2.Sobel(
        normalized,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    grad_y = cv2.Sobel(
        normalized,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    light_angle = random.uniform(
        0,
        2 * np.pi
    )

    light_x = np.cos(light_angle)
    light_y = np.sin(light_angle)

    relief = (
        grad_x * light_x
        + grad_y * light_y
    )

    carved += (
        np.maximum(relief, 0)
        * random.uniform(8, 18)
    )

    carved -= (
        np.maximum(-relief, 0)
        * random.uniform(10, 22)
    )

    # Local noise inside groove.
    groove_noise = np.random.normal(
        0,
        random.uniform(1.0, 3.5),
        (h, w)
    ).astype(np.float32)

    carved += (
        groove_noise
        * normalized
    )

    image[
        y:y + h,
        x:x + w
    ] = carved

    return image


# ============================================================
# 9. BOX OVERLAP CHECK
# ============================================================

def boxes_overlap(
    box_a,
    box_b,
    gap=MIN_BOX_GAP
):
    """
    Check whether two boxes overlap or come
    too close to one another.

    Box format:
        x1, y1, x2, y2
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    return not (
        ax2 + gap < bx1
        or bx2 + gap < ax1
        or ay2 + gap < by1
        or by2 + gap < ay1
    )


# ============================================================
# 10. PLACE ONE SYMBOL
# ============================================================

def place_symbol(
    image,
    mask,
    existing_boxes,
    tier
):
    """
    Try to place a glyph somewhere on the stone.

    Returns:
        (image, bbox)
        or
        (None, None)
    """

    h, w = mask.shape

    if h >= IMAGE_HEIGHT or w >= IMAGE_WIDTH:
        return None, None

    for _ in range(
        MAX_PLACEMENT_ATTEMPTS
    ):

        x = random.randint(
            5,
            IMAGE_WIDTH - w - 5
        )

        y = random.randint(
            5,
            IMAGE_HEIGHT - h - 5
        )

        bbox = (
            x,
            y,
            x + w,
            y + h
        )

        if any(
            boxes_overlap(
                bbox,
                other
            )
            for other in existing_boxes
        ):
            continue

        image = carve_glyph(
            image,
            mask,
            x,
            y,
            tier
        )

        return image, bbox

    return None, None


# ============================================================
# 11. FINAL IMAGE DEGRADATION
# ============================================================

def apply_image_degradation(
    image,
    tier
):
    """
    Apply image-level effects after all symbols
    have been placed.
    """

    image = image.astype(
        np.float32
    )

    # Exposure / contrast
    alpha, beta = {
        "clean": (
            random.uniform(0.96, 1.05),
            random.uniform(-3, 3)
        ),
        "moderate": (
            random.uniform(0.90, 1.10),
            random.uniform(-8, 8)
        ),
        "heavy": (
            random.uniform(0.82, 1.18),
            random.uniform(-14, 14)
        ),
    }[tier]

    image = image * alpha + beta

    # Blur
    blur_probability = {
        "clean": 0.15,
        "moderate": 0.35,
        "heavy": 0.55,
    }[tier]

    if random.random() < blur_probability:

        sigma = random.uniform(
            0.25,
            0.80
        )

        image = cv2.GaussianBlur(
            image,
            (3, 3),
            sigma
        )

    # Camera-like noise
    noise_std = {
        "clean": random.uniform(1.0, 2.0),
        "moderate": random.uniform(1.5, 3.5),
        "heavy": random.uniform(2.5, 5.5),
    }[tier]

    image += np.random.normal(
        0,
        noise_std,
        image.shape
    )

    # Mild vignette
    if random.random() < 0.30:

        yy, xx = np.mgrid[
            0:IMAGE_HEIGHT,
            0:IMAGE_WIDTH
        ]

        cx = IMAGE_WIDTH / 2
        cy = IMAGE_HEIGHT / 2

        distance = np.sqrt(
            ((xx - cx) / cx) ** 2
            + ((yy - cy) / cy) ** 2
        )

        vignette = (
            1.0
            - np.clip(distance, 0, 1)
            * random.uniform(0.03, 0.12)
        )

        image *= vignette

    return np.clip(
        image,
        25,
        230
    ).astype(np.uint8)


# ============================================================
# 12. CREATE ONE SYNTHETIC IMAGE
# ============================================================

def generate_image(
    characters
):
    """
    Generate one complete synthetic stone image
    and its YOLO annotations.

    Returns:
        image,
        annotations

    annotations:
        [
            (
                class_id,
                x_center,
                y_center,
                width,
                height
            ),
            ...
        ]
    """

    tier = choose_tier()

    image = create_stone_background(
        tier
    )

    existing_boxes = []
    annotations = []

    number_of_symbols = random.randint(
        MIN_SYMBOLS_PER_IMAGE,
        MAX_SYMBOLS_PER_IMAGE
    )

    for _ in range(
        number_of_symbols
    ):

        label = random.choice(
            CLASS_NAMES
        )

        char = characters[label]

        mask = render_glyph_mask(
            char,
            tier
        )

        if mask is None:
            continue

        mask = transform_glyph(
            mask,
            tier
        )

        result = place_symbol(
            image,
            mask,
            existing_boxes,
            tier
        )

        if result is None or result[0] is None:
            continue

        image, bbox = result

        x1, y1, x2, y2 = bbox

        existing_boxes.append(
            bbox
        )

        # YOLO normalized coordinates.
        center_x = (
            (x1 + x2) / 2
        ) / IMAGE_WIDTH

        center_y = (
            (y1 + y2) / 2
        ) / IMAGE_HEIGHT

        box_width = (
            x2 - x1
        ) / IMAGE_WIDTH

        box_height = (
            y2 - y1
        ) / IMAGE_HEIGHT

        class_id = CLASS_NAMES.index(
            label
        )

        annotations.append(
            (
                class_id,
                center_x,
                center_y,
                box_width,
                box_height
            )
        )

    image = apply_image_degradation(
        image,
        tier
    )

    return image, annotations


# ============================================================
# 13. SAVE YOLO LABEL FILE
# ============================================================

def save_yolo_labels(
    path,
    annotations
):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for (
            class_id,
            center_x,
            center_y,
            width,
            height
        ) in annotations:

            f.write(
                f"{class_id} "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )


# ============================================================
# 14. PREPARE DIRECTORIES
# ============================================================

def prepare_split(
    split
):
    image_dir = os.path.join(
        OUTPUT_DIR,
        "images",
        split
    )

    label_dir = os.path.join(
        OUTPUT_DIR,
        "labels",
        split
    )

    os.makedirs(
        image_dir,
        exist_ok=True
    )

    os.makedirs(
        label_dir,
        exist_ok=True
    )

    if OVERWRITE:

        for filename in os.listdir(
            image_dir
        ):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                os.remove(
                    os.path.join(
                        image_dir,
                        filename
                    )
                )

        for filename in os.listdir(
            label_dir
        ):

            if filename.lower().endswith(
                ".txt"
            ):

                os.remove(
                    os.path.join(
                        label_dir,
                        filename
                    )
                )

    return image_dir, label_dir


# ============================================================
# 15. GENERATE SPLIT
# ============================================================

def generate_split(
    split,
    count,
    characters
):

    image_dir, label_dir = (
        prepare_split(split)
    )

    print(
        f"\nGenerating {split}: "
        f"{count} images"
    )

    generated = 0

    while generated < count:

        image, annotations = (
            generate_image(
                characters
            )
        )

        # Never save an image without annotations.
        if not annotations:
            continue

        filename = (
            f"synthetic_{generated:06d}"
        )

        image_path = os.path.join(
            image_dir,
            filename + ".jpg"
        )

        label_path = os.path.join(
            label_dir,
            filename + ".txt"
        )

        cv2.imwrite(
            image_path,
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        save_yolo_labels(
            label_path,
            annotations
        )

        generated += 1

        if generated % 100 == 0:
            print(
                f"  {generated}/{count}",
                flush=True
            )


# ============================================================
# 16. DATASET YAML
# ============================================================

def save_dataset_yaml():
    yaml_path = os.path.join(
        OUTPUT_DIR,
        "dataset.yaml"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            "names:\n"
        )

        for class_id, name in enumerate(
            CLASS_NAMES
        ):

            f.write(
                f"  {class_id}: {name}\n"
            )

    print(
        f"\nYOLO configuration saved: "
        f"{yaml_path}"
    )


# ============================================================
# 17. MAIN
# ============================================================

def main():

    if not os.path.exists(
        FONT_PATH
    ):

        print(
            "\nERROR: Font not found:"
        )

        print(
            f"  {FONT_PATH}"
        )

        return

    print(
        "Discovering Cypriot "
        "Syllabary characters..."
    )

    characters = (
        get_cypriot_characters()
    )

    print(
        f"Found {len(characters)} "
        "Cypriot syllabary classes."
    )

    print(
        "\nClass mapping:"
    )

    for class_id, label in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {class_id:2d} -> {label}"
        )

    generate_split(
        "train",
        TRAIN_IMAGES,
        characters
    )

    generate_split(
        "val",
        VAL_IMAGES,
        characters
    )

    generate_split(
        "test",
        TEST_IMAGES,
        characters
    )

    save_dataset_yaml()

    print(
        "\n================================"
    )

    print(
        "Synthetic YOLO dataset complete."
    )

    print(
        "================================"
    )

    print(
        f"\nOutput:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()