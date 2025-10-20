from typing import Any, Dict, List, Tuple
from collections import defaultdict
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import utils

# --------------------------------------------------------------------------
# Academic Honesty Policy
# --------------------------------------------------------------------------

# The below credentials are equivalent to signing the academic honesty policy
STUDENT_NAME = "Ariel Ben Avi"  # TODO: Fill in your name
STUDENT_UNI = "ab5884"  # TODO: Fill in your UNI


def sign_academic_honesty_policy():
    assert (
        STUDENT_NAME is not None and STUDENT_UNI is not None
    ), "Please fill in your STUDENT_NAME and STUDENT_UNI at the top of solutions_UNI.py"
    print(
        f"I, {STUDENT_NAME} ({STUDENT_UNI}), certify that I have read and agree to the"
        " Code of Academic Integrity."
    )


# --------------------------------------------------------------------------
# Walkthrough 1: Image processing
# --------------------------------------------------------------------------
def walkthrough1a():
    """
    Image processing: convolution, Gaussian smoothing

    Image credit: http://commons.wikimedia.org/wiki/File:Beautiful-pink-flower_-_West_Virginia_-_ForestWander.jpg
    """
    # Load the image
    img = utils.imread(utils.get_data_path("flower.png"))

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the sigma values
    sigma = [6, 12, 24]

    # Define the kernel sizes
    k = [int(np.ceil(2 * np.pi * s)) // 2 * 2 + 1 for s in sigma]

    # Create a figure with 2 rows and 2 columns of subplots
    fig, axs = plt.subplots(2, 2)

    # Display the original image
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Original")

    # Loop through the sigma values and apply Gaussian blur
    for i in range(len(sigma)):
        # Generate a Gaussian kernel
        h = cv2.GaussianBlur(img_gray, (k[i], k[i]), sigma[i])

        # Display the blurred image
        axs[(i + 1) // 2, (i + 1) % 2].imshow(h, cmap="gray")
        axs[(i + 1) // 2, (i + 1) % 2].axis("off")
        axs[(i + 1) // 2, (i + 1) % 2].set_title(f"sigma = {sigma[i]}")

    # Set the plot layout and save the figure
    fig.tight_layout()
    plt.savefig(utils.get_result_path("w1a_blur-flowers.png"))

    # Show result plot
    plt.show()


def walkthrough1b():
    """
    Edge detection

    Image credit: CAVE Lab
    """
    # Load the image
    img = utils.imread(utils.get_data_path("hello.png"))

    # Display the color image
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Color Image")

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    axs[0, 1].imshow(gray_img, cmap="gray")
    axs[0, 1].axis("off")
    axs[0, 1].set_title("Grayscale Image")

    # Sobel edge detection
    thresh = 0.05 * np.max(gray_img)
    edge_img = cv2.Sobel(gray_img, cv2.CV_64F, 1, 1, ksize=3)
    edge_img[edge_img < thresh] = 0
    edge_img[edge_img >= thresh] = 255
    edge_img = edge_img.astype(np.uint8)
    axs[1, 0].imshow(edge_img, cmap="gray")
    axs[1, 0].axis("off")
    axs[1, 0].set_title("Sobel Edge Detection")

    # Canny edge detection
    thresh1 = 0.05 * np.max(gray_img)
    thresh2 = 0.1 * np.max(gray_img)
    edge_img = cv2.Canny(gray_img, thresh1, thresh2)
    axs[1, 1].imshow(edge_img, cmap="gray")
    axs[1, 1].axis("off")
    axs[1, 1].set_title("Canny Edge Detection")

    # Save the resulting image
    plt.savefig(utils.get_result_path("w1b_hello-edges.png"))

    # Show result plot
    plt.show()


# --------------------------------------------------------------------------
# Challenge 1: Line Detection
# --------------------------------------------------------------------------
def find_edge_pixels(img: np.ndarray) -> np.ndarray:
    """
    Use the Canny edge detection function from OpenCV

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The edge image returned by Canny.
    """
    thresh1 = 100
    thresh2 = 200
    
    edge_img = cv2.Canny(img, thresh1, thresh2)
    
    return edge_img
    # raise NotImplementedError()

def generate_hough_accumulator(edge_img: np.ndarray) -> np.ndarray:
    """
    make the hough accumulator. detect lines from edge image.
    """

    # get height and width of the edge image
    h, w = edge_img.shape

    # max distance from origin (corner to corner)
    max_rho = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))

    # theta go from -90 to +90 degree (in rad)
    thetas = np.deg2rad(np.arange(-90, 90))

    # pre compute cos and sin for speed
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # make empty vote table, rho can be negative so use 2*max_rho
    acc = np.zeros((2 * max_rho, len(thetas)), dtype=np.uint64)

    # find where edge pixels exist (white pixel = 255)
    y_idx, x_idx = np.nonzero(edge_img)

    # go over every edge pixel and vote
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]

        # each pixel vote for many (rho, theta)
        for t_i in range(len(thetas)):
            rho = int(round(x * cos_t[t_i] + y * sin_t[t_i]))
            acc[rho + max_rho, t_i] += 1  # shift rho index so no negative index

    # normalize so we can see it like image (0-255)
    hough_img = cv2.normalize(np.log1p(acc), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # plt.imshow(hough_img, cmap='hot', aspect='auto')
    # plt.title("Hough accumulator")
    # plt.xlabel("theta index")
    # plt.ylabel("rho index")
    # plt.show()

    # print("done hough accumulator:", hough_img.shape)

    return hough_img



def line_finder(orig_img: np.ndarray, hough_img: np.ndarray) -> np.ndarray:
    """
    Find the lines in the original image using the Hough accumulator.

    Args:
        orig_img (np.ndarray): The original image to find lines in.
        hough_img (np.ndarray): The Hough accumulator.

    Returns:
        np.ndarray: The original image overlaid with lines.

    # METHOD EXPLANATION:
    # Use a simple threshold + non-maximum suppression (NMS) approach to find "strong" lines
    # (1) normalize + threshold (keep peaks above 0.88*max)
    # (2) NMS via dilation to get local maxima
    # (3) sort by strength; keep top-K
    # (4) convert (rho, theta) to image lines
    """

	# copy image to not destroy original
    out = orig_img.copy()

	# recover mapping from accumulator index -> (rho, theta)
    H, W = orig_img.shape[:2]

    # max possible rho from this image (corner to corner)
    max_rho = int(np.ceil(np.sqrt(H * H + W * W)))

    num_rho_bins, num_theta_bins = hough_img.shape[:2]

    # -90 < theta < 90
    theta_vals = np.deg2rad(np.linspace(-90.0, 90.0, num=num_theta_bins, endpoint=False))

    rho_vals = np.linspace(-max_rho, max_rho, num=num_rho_bins, endpoint=False)

    # work with float accumulator for thresholding (uint8 image from 1b)
    acc = hough_img.astype(np.float32)

    # pick peaks in accumulator (threshold + non-maximum suppression)
    # threshold relative to max. if image very clean this is ok.
    acc_max = float(acc.max()) if acc.size else 0.0
    if acc_max <= 0:
        return out  # nothing to draw

    thr = 0.88 * acc_max

    k = np.ones((15, 15), np.uint8)
    local_max = acc == cv2.dilate(acc, k)

    # candidates = above threshold and local peak
    cand = np.where((acc >= thr) & local_max)

    # collect peaks with their score
    peaks = [(int(r), int(c), float(acc[r, c])) for r, c in zip(cand[0], cand[1])]

    peaks.sort(key=lambda x: x[2], reverse=True)
    top_k = 6  # tune number of lines to draw
    peaks = peaks[:top_k]

    # ----- 3) draw each detected line on the image -----
    for r_idx, t_idx, _score in peaks:
        rho = float(rho_vals[r_idx])
        theta = float(theta_vals[t_idx])

        # line equation: x*cos(theta) + y*sin(theta) = rho
        a = np.cos(theta)
        b = np.sin(theta)

        # point on line closest to origin
        x0 = a * rho
        y0 = b * rho

        # direction vector perpendicular to (a, b) is (-b, a)
        # choose big length so it crosses the image for sure
        L = 2000.0
        x1 = int(round(x0 + L * (-b)))
        y1 = int(round(y0 + L * (a)))
        x2 = int(round(x0 - L * (-b)))
        y2 = int(round(y0 - L * (a)))

        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    return out



# --------------------------------------------------------------------------
# Challenge 2: Coin Counting
# --------------------------------------------------------------------------


def _edge_gray(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	# helper: gray + edges (canny). return (gray, edge)
	if img_bgr.ndim == 3:
		gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
	else:
		gray = img_bgr.copy()
	# a little blur so edges not too noisy
	gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
	edges = cv2.Canny(gray_blur, 80, 160)
	return gray, edges


def generate_coin_hough_accumulator_2b(
    orig_img: np.ndarray,
    coins_info: Dict[int, Dict[str, Any]],
    radius_range: int = 3,
) -> Dict[Tuple[int, int, int], int]:
    """
    Generate the Circle Hough accumulators for coin detection

    Args:
        orig_img (np.ndarray): The original image to find coins in.
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        radius_range (int, optional): The range of radius to search for. Defaults to 0.

    Returns:
        Dict[Tuple[int, int, int], int]: The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.
    """
    H, W = orig_img.shape[:2]
    _, edges = _edge_gray(orig_img)

    # radii come from coins_info keys
    base_radii = sorted(list(coins_info.keys()))

    # angle step: 360/90 = 4 deg steps -> ok speed/accuracy trade
    thetas = np.deg2rad(np.arange(0, 360, 2))
    ct = np.cos(thetas)
    st = np.sin(thetas)

    acc = defaultdict(int)

    # sample edges a bit (every 1 px) to speed
    ys, xs = np.where(edges > 0)
    for idx in range(0, len(xs), 1):
        x = int(xs[idx])
        y = int(ys[idx])

        for r0 in base_radii:
            # search small band around radius if ask radius_range
            r_min = max(1, r0 - radius_range)
            r_max = r0 + radius_range
            for r in range(r_min, r_max + 1):
                # compute centers around circle for this radius
                ax = x - (r * ct)
                by = y - (r * st)
                ax = np.rint(ax).astype(np.int32)
                by = np.rint(by).astype(np.int32)

                # vote if center inside image
                inside = (ax >= 0) & (ax < W) & (by >= 0) & (by < H)
                for a, b, ok in zip(ax, by, inside):
                    if ok:
                        acc[(int(a), int(b), int(r))] += 1

    return dict(acc)


# helper: merge duplicate detections across radii (keep strongest vote)
def _merge_coins(dets: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
	dets.sort(key=lambda t: t[3], reverse=True)  # by votes
	kept: List[Tuple[int,int,int,int]] = []
	for cx, cy, r, v in dets:
		ok = True
		for kx, ky, kr, kv in kept:
			# if centers very close, treat as same coin (radius can differ)
			if (cx - kx) ** 2 + (cy - ky) ** 2 < (0.5 * max(r, kr)) ** 2:
				ok = False
				break
		if ok:
			kept.append((cx, cy, r, v))
	return kept


def coin_finder_2b(
    coin_hough: Dict[Tuple[int, int, int], int]
) -> List[Tuple[int, int, int, int]]:
    """
    find coins from accumulator dict
    output list = (cx, cy, r, votes)
    """
    if not coin_hough:
        return []

    max_x = max(k[0] for k in coin_hough.keys())
    max_y = max(k[1] for k in coin_hough.keys())
    H = max_y + 1
    W = max_x + 1

    from collections import defaultdict as dd
    by_r: Dict[int, List[Tuple[Tuple[int, int, int], int]]] = dd(list)
    for k, v in coin_hough.items():
        by_r[k[2]].append((k, v))

    all_dets: List[Tuple[int, int, int, int]] = []

    for r, items in by_r.items():
        A = np.zeros((H, W), dtype=np.float32)
        for (cx, cy, _), v in items:
            if 0 <= cy < H and 0 <= cx < W:
                A[cy, cx] += float(v)

        mx = float(A.max())
        if mx <= 0:
            continue

        # tighter, radius-aware threshold
        if r <= 85:       # dime
            thr = 0.70 * mx
        elif r <= 105:    # nickel
            thr = 0.75 * mx
        else:             # quarter
            thr = 0.78 * mx

        # stronger NMS
        k = np.ones((21, 21), np.uint8)
        local_max = A == cv2.dilate(A, k)
        mask = (A >= thr) & local_max

        ys, xs = np.where(mask)
        peaks = [(int(xs[i]), int(ys[i]), int(r), int(A[ys[i], xs[i]])) for i in range(len(xs))]
        peaks.sort(key=lambda t: t[3], reverse=True)

        # prune centers too close within same radius
        min_dist = max(8, int(0.9 * r))
        final_r: List[Tuple[int, int, int, int]] = []
        for cx, cy, rr, vv in peaks:
            ok = True
            for px, py, _, _ in final_r:
                if (cx - px) ** 2 + (cy - py) ** 2 < (min_dist ** 2):
                    ok = False
                    break
            if ok:
                final_r.append((cx, cy, rr, vv))
            if len(final_r) >= 50:
                break

        all_dets.extend(final_r)

    # merge duplicates across radii
    all_dets = _merge_coins(all_dets)

    return all_dets


def get_total_value_2b(
    coins_info: Dict[int, Dict[str, Any]],
    detected_coins: List[Tuple[int, int, int, int]],
    radius_range: int = 10,
) -> float:
    """
    Find the total value of all the detected coins

    Args:
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        detected_coins (List[Tuple[int, int, int, int]]): Information of the detected coins -- a
            list of (x, y, r, votes).
        radius_range (int, optional): The range of radius to search for. Defaults to 10.

    Returns:
        float: The total value in USD.
    """
    if not detected_coins:
        return 0.0

    known = sorted(coins_info.keys())
    total_cents = 0
    for x, y, r, _ in detected_coins:
        # nearest radius from known
        best_r = min(known, key=lambda rr: abs(rr - r))
        if abs(best_r - r) <= radius_range:
            total_cents += int(coins_info[best_r]["value"])

    return round(total_cents / 100.0, 2)


def generate_coin_hough_accumulator_2c(
    orig_img: np.ndarray,
    coins_info: Dict[int, Dict[str, Any]],
    radius_range: int = 4,
) -> Dict[Tuple[int, int, int], int]:
    """
    Generate the Circle Hough accumulators for coin detection

    Args:
        orig_img (np.ndarray): The original image to find coins in.
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        radius_range (int, optional): The range of radius to search for. Defaults to 0.

    Returns:
        Dict[Tuple[int, int, int], int]: The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.
    """
    H, W = orig_img.shape[:2]
    if orig_img.ndim == 3:
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = orig_img.copy()

    # stronger blur to merge tiny gaps
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 1.8)
    edges = cv2.Canny(gray_blur, 90, 180)

    base_radii = sorted(list(coins_info.keys()))
    thetas = np.deg2rad(np.arange(0, 360, 3))  # denser angles
    ct = np.cos(thetas)
    st = np.sin(thetas)

    acc = defaultdict(int)
    ys, xs = np.where(edges > 0)

    for idx in range(0, len(xs), 1):  # use all edge pixels here
        x = int(xs[idx]); y = int(ys[idx])
        for r0 in base_radii:
            r_min = max(1, r0 - radius_range)
            r_max = r0 + radius_range
            for r in range(r_min, r_max + 1):
                ax = x - (r * ct)
                by = y - (r * st)
                ax = np.rint(ax).astype(np.int32)
                by = np.rint(by).astype(np.int32)
                inside = (ax >= 0) & (ax < W) & (by >= 0) & (by < H)
                for a, b, ok in zip(ax, by, inside):
                    if ok:
                        acc[(int(a), int(b), int(r))] += 1

    return dict(acc)


def coin_finder_2c(
    coin_hough: Dict[Tuple[int, int, int], int]
) -> List[Tuple[int, int, int, int]]:
    """
    Find the coins using the Circle Hough accumulator.

    Args:
        coin_hough (Dict[Tuple[int, int, int], int]): The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.

    Returns:
        List[Tuple[int, int, int, int]]: Information of the detected coins -- a list of (x, y, r, votes).
    """
    if not coin_hough:
        return []

    max_x = max(k[0] for k in coin_hough.keys())
    max_y = max(k[1] for k in coin_hough.keys())
    H = max_y + 1
    W = max_x + 1

    from collections import defaultdict as dd
    by_r: Dict[int, List[Tuple[Tuple[int, int, int], int]]] = dd(list)
    for k, v in coin_hough.items():
        by_r[k[2]].append((k, v))

    all_dets: List[Tuple[int, int, int, int]] = []

    for r, items in by_r.items():
        A = np.zeros((H, W), dtype=np.float32)
        for (cx, cy, _), v in items:
            if 0 <= cy < H and 0 <= cx < W:
                A[cy, cx] += float(v)

        mx = float(A.max())
        if mx <= 0:
            continue

        # tougher thresholds for touching coins
        if r <= 85:        # dime
            thr = 0.75 * mx
        elif r <= 105:     # nickel
            thr = 0.78 * mx
        else:              # quarter
            thr = 0.80 * mx

        # bigger NMS window to collapse clusters
        k = np.ones((23, 23), np.uint8)
        local_max = (A == cv2.dilate(A, k))
        mask = (A >= thr) & local_max

        ys, xs = np.where(mask)
        peaks = [(int(xs[i]), int(ys[i]), int(r), int(A[ys[i], xs[i]])) for i in range(len(xs))]
        peaks.sort(key=lambda t: t[3], reverse=True)

        # enforce larger separation (about 1.0*r)
        min_dist = max(10, int(1.0 * r))
        final_r: List[Tuple[int, int, int, int]] = []
        for cx, cy, rr, vv in peaks:
            ok = True
            for px, py, _, _ in final_r:
                if (cx - px) ** 2 + (cy - py) ** 2 < (min_dist ** 2):
                    ok = False
                    break
            if ok:
                final_r.append((cx, cy, rr, vv))
            if len(final_r) >= 60:
                break

        all_dets.extend(final_r)

    # merge duplicates across radii (helper defined in 2b)
    all_dets = _merge_coins(all_dets)

    return all_dets


def get_total_value_2c(
    coins_info: Dict[int, Dict[str, Any]],
    detected_coins: List[Tuple[int, int, int, int]],
    radius_range: int = 10,
) -> float:
    """
    Find the total value of all the detected coins

    Args:
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        detected_coins (List[Tuple[int, int, int, int]]): Information of the detected coins -- a
            list of (x, y, r, votes).
        radius_range (int, optional): The range of radius to search for. Defaults to 10.

    Returns:
        float: The total value in USD.
    """
    # same as 2b
    return get_total_value_2b(coins_info, detected_coins, radius_range)



def generate_coin_hough_accumulator_2d(
    orig_img: np.ndarray,
    coins_info: Dict[int, Dict[str, Any]],
    radius_range: int = 5,
) -> Dict[Tuple[int, int, int], int]:
    """
    Generate the Circle Hough accumulators for coin detection

    Args:
        orig_img (np.ndarray): The original image to find coins in.
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        radius_range (int, optional): The range of radius to search for. Defaults to 0.

    Returns:
        Dict[Tuple[int, int, int], int]: The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.
    """
    if orig_img.ndim == 3:
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = orig_img.copy()
    blur = cv2.GaussianBlur(gray, (9, 9), 2.2)

    # --- foreground mask to remove text/background ---
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ensure coins are white in mask
    if np.mean(blur[mask > 0]) < np.mean(blur[mask == 0]):
        mask = cv2.bitwise_not(mask)
    k_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_fg, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_fg, iterations=1)

    # --- edges (masked) ---
    edges = cv2.Canny(blur, 100, 200)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    # --- gradient dir for fast voting (two normals only) ---
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    phi = np.arctan2(gy, gx)
    mag = cv2.magnitude(gx, gy)
    if float(mag.max()) > 0:
        mag /= (mag.max() + 1e-6)

    edge_mag_mask = ((mag > 0.20).astype(np.uint8)) * 255
    edges = cv2.bitwise_and(edges, edges, mask=edge_mag_mask)

    H, W = gray.shape[:2]
    base_radii = sorted(list(coins_info.keys()))
    acc = defaultdict(int)

    ys, xs = np.where(edges > 0)
    for i in range(len(xs)):
        x = int(xs[i]); y = int(ys[i])
        ang = float(phi[y, x])
        ca, sa = math.cos(ang), math.sin(ang)
        w = 1 + (1 if mag[y, x] > 0.35 else 0)  # small weight for strong edges

        for r0 in base_radii:
            r_min = max(1, r0 - radius_range)
            r_max = r0 + radius_range
            for r in range(r_min, r_max + 1):
                cx1 = int(round(x + r * ca)); cy1 = int(round(y + r * sa))
                cx2 = int(round(x - r * ca)); cy2 = int(round(y - r * sa))
                if 0 <= cx1 < W and 0 <= cy1 < H: acc[(cx1, cy1, r)] += w
                if 0 <= cx2 < W and 0 <= cy2 < H: acc[(cx2, cy2, r)] += w

    return dict(acc)


def coin_finder_2d(
    coin_hough: Dict[Tuple[int, int, int], int]
) -> List[Tuple[int, int, int, int]]:
    """
    Find the coins using the Circle Hough accumulator.

    Args:
        coin_hough (Dict[Tuple[int, int, int], int]): The Circle Hough accumulator. A dictionary mapping the
            (centroid_x, centroid_y, radius) to the number of votes.

    Returns:
        List[Tuple[int, int, int, int]]: Information of the detected coins -- a list of (x, y, r, votes).
    """
    if not coin_hough:
        return []

    max_x = max(k[0] for k in coin_hough.keys())
    max_y = max(k[1] for k in coin_hough.keys())
    H = max_y + 1; W = max_x + 1

    from collections import defaultdict as dd
    by_r: Dict[int, List[Tuple[Tuple[int, int, int], int]]] = dd(list)
    for k, v in coin_hough.items():
        by_r[k[2]].append((k, v))

    all_dets: List[Tuple[int, int, int, int]] = []

    for r, items in by_r.items():
        A = np.zeros((H, W), dtype=np.float32)
        for (cx, cy, _), v in items:
            if 0 <= cy < H and 0 <= cx < W:
                A[cy, cx] += float(v)

        mx = float(A.max())
        if mx <= 0: 
            continue

        # softer, radius-aware thresholds (noisy background)
        if r <= 85:      thr = 0.54 * mx   # dime
        elif r <= 105:   thr = 0.58 * mx   # nickel
        else:            thr = 0.64 * mx   # quarter

        # strong NMS
        k = np.ones((23, 23), np.uint8)
        local_max = (A == cv2.dilate(A, k))
        mask = (A >= thr) & local_max

        ys, xs = np.where(mask)
        peaks = [(int(xs[i]), int(ys[i]), int(r), int(A[ys[i], xs[i]])) for i in range(len(xs))]
        peaks.sort(key=lambda t: t[3], reverse=True)

        # keep well-spaced centers
        min_dist = max(10, int(0.95 * r))
        chosen: List[Tuple[int, int, int, int]] = []
        for cx, cy, rr, vv in peaks:
            ok = True
            for px, py, _, _ in chosen:
                if (cx - px) ** 2 + (cy - py) ** 2 < (min_dist ** 2):
                    ok = False; break
            if ok:
                chosen.append((cx, cy, rr, vv))
            if len(chosen) >= 80:
                break

        all_dets.extend(chosen)
    
    return _merge_coins(all_dets)


def get_total_value_2d(
    coins_info: Dict[int, Dict[str, Any]],
    detected_coins: List[Tuple[int, int, int, int]],
    radius_range: int = 10,
) -> float:
    """
    Find the total value of all the detected coins

    Args:
        coins_info (Dict[int, Dict[str, Any]]): The dictionary containing the information
            of coins.
        detected_coins (List[Tuple[int, int, int, int]]): Information of the detected coins -- a
            list of (x, y, r, votes).
        radius_range (int, optional): The range of radius to search for. Defaults to 10.

    Returns:
        float: The total value in USD.
    """
    # same as 2b
    return get_total_value_2b(coins_info, detected_coins, radius_range)
