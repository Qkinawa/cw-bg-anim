from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter, label
from skimage.graph import MCP, MCP_Geometric
from skimage.segmentation import slic

# у каждого эффекта свой поток рандома, чтоб правка одной зоны не сбивала другие
_RNG_STREAM = {"light": 11, "dust": 12, "eyes": 13,
               "water": 1, "plants": 2, "mist": 3, "fireflies": 4, "glow": 5, "rays": 6,
               "particles": 7, "clouds": 8, "wind": 9, "flap": 10}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _as_list(v):
    if not v:
        return []
    return list(v) if isinstance(v, list) else [v]


def load_defaults(path):
    return json.loads(Path(path).read_text("utf-8"))


# конфиг локации + дефолты для каждого эффекта
def load_config(path, defaults):
    raw = json.loads(Path(path).read_text("utf-8"))
    eff = defaults["effects"]
    cfg = deep_merge({k: v for k, v in defaults.items() if k != "effects"},
                     {k: v for k, v in raw.items() if k not in eff})
    for key in eff:
        # выключенные зоны не сдвигают номера остальных, иначе у соседей поменяется рандом
        zones = [(k, z) for k, z in enumerate(_as_list(raw.get(key))) if z.get("enabled", True)]
        cfg[key] = [deep_merge(eff[key], {"label": f"{key}{k + 1}", "_idx": k, **z})
                    for k, z in zones]
    return cfg


def _pts(points):
    return np.round(np.asarray(points, np.float64)).astype(np.int32).reshape(-1, 2)


def polygon_mask(points, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [_pts(points)], 1)
    return m.astype(bool)


def _polys(v):
    if not v:
        return []
    return [v] if isinstance(v[0][0], (int, float)) else list(v)


def polys_mask(v, shape):
    m = np.zeros(shape, bool)
    for p in _polys(v):
        m |= polygon_mask(p, shape)
    return m


# точка или линия корня, там растение не качается
def root_mask(root, shape):
    H, W = shape
    p = _pts(root)
    p[:, 0] = np.clip(p[:, 0], 0, W - 1)
    p[:, 1] = np.clip(p[:, 1], 0, H - 1)
    m = np.zeros(shape, np.uint8)
    if len(p) == 1:
        cv2.circle(m, (int(p[0, 0]), int(p[0, 1])), 1, 1, -1)
    else:
        cv2.polylines(m, [p], False, 1, 2)
    return m.astype(bool)


def rgb_to_hsv(img):
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2HSV)


# достаёт значение для цветового условия: h, s, v, r, g, b, lum или разницу каналов типа g-r
def _channel(sc, key):
    if key in ("h", "s", "v"):
        return sc.hsv[..., "hsv".index(key)]
    if key.startswith("|") and key.endswith("|"):
        return np.abs(_channel(sc, key[1:-1]))
    if "-" in key:
        a, b = key.split("-")
        return _channel(sc, a) - _channel(sc, b)
    if key == "lum":
        return sc.lum
    return sc.img[..., "rgb".index(key)]


# какие пиксели подходят под цвета из конфига
def color_match(sc, ranges):
    out = np.zeros(sc.shape, bool)
    for r in _as_list(ranges):
        m = np.ones(sc.shape, bool)
        for key, (lo, hi) in r.items():
            v = _channel(sc, key)
            m &= ((v >= lo) & (v <= hi)) if (lo <= hi or key != "h") else ((v >= lo) | (v <= hi))
        out |= m
    return out


_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def dilate(m, it):
    if it <= 0:
        return m.copy()
    return cv2.dilate(m.astype(np.uint8), _CROSS, iterations=int(it)).astype(bool)


# расширение кругом, крестом получается ромб
def dilate_round(m, r):
    r = int(r)
    if r <= 0:
        return m.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k).astype(bool)


def erode(m, it):
    if it <= 0:
        return m.copy()
    return cv2.erode(m.astype(np.uint8), _CROSS, iterations=int(it)).astype(bool)


def morph(m, op, it):
    if it <= 0:
        return m.copy()
    return cv2.morphologyEx(m.astype(np.uint8), op, _CROSS, iterations=int(it)).astype(bool)


def remove_small(m, min_area):
    if min_area <= 0 or not m.any():
        return m
    lab, _ = label(m)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return np.isin(lab, np.nonzero(sizes >= min_area)[0])


# плавно гасит эффект к краю зоны, чтоб не было резкой границы
def edge_falloff(zone, feather):
    if feather <= 0:
        return zone.astype(np.float32)
    f = int(math.ceil(feather))
    pad = cv2.copyMakeBorder(zone.astype(np.uint8), f, f, f, f, cv2.BORDER_REPLICATE)
    d = cv2.distanceTransform(pad, cv2.DIST_L2, 5)[f:-f, f:-f]
    t = np.clip(d / feather, 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def clean_mask(m, p):
    m = morph(m, cv2.MORPH_OPEN, p["open_iter"])
    m = morph(m, cv2.MORPH_CLOSE, p["close_iter"])
    return remove_small(m, p["min_area"])


# мягкий край маски, который липнет к краям самой картинки
def guided_filter(I, p, r, eps):
    k = (2 * r + 1, 2 * r + 1)
    mean = lambda x: cv2.boxFilter(x, -1, k)
    mI, mp = mean(I), mean(p)
    a = (mean(I * p) - mI * mp) / (mean(I * I) - mI * mI + eps)
    b = mp - a * mI
    return np.clip(mean(a) * I + mean(b), 0, 1).astype(np.float32)


def _bbox(m, margin, shape):
    ys, xs = np.nonzero(m)
    H, W = shape
    return (max(0, xs.min() - margin), max(0, ys.min() - margin),
            min(W, xs.max() + margin + 1), min(H, ys.max() + margin + 1))


# маска через grabcut, цвета из конфига идут как подсказка
def _grabcut(sc, zone, seed, fg, bg, z):
    x0, y0, x1, y1 = _bbox(zone, int(z["grabcut_margin"]), sc.shape)
    m = np.full(sc.shape, cv2.GC_BGD, np.uint8)
    if seed is not None and seed.any():
        m[zone] = cv2.GC_PR_BGD
        m[seed & zone] = cv2.GC_PR_FGD
    else:
        m[zone] = cv2.GC_PR_FGD
    m[fg & zone] = cv2.GC_FGD
    m[bg] = cv2.GC_BGD
    sub = np.ascontiguousarray(m[y0:y1, x0:x1])
    if not ((sub == cv2.GC_FGD) | (sub == cv2.GC_PR_FGD)).any():
        return np.zeros(sc.shape, bool)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    cv2.grabCut(np.ascontiguousarray(sc.u8[y0:y1, x0:x1, ::-1]), sub, None, bgd, fgd,
                int(z["grabcut_iters"]), cv2.GC_INIT_WITH_MASK)
    out = np.zeros(sc.shape, bool)
    out[y0:y1, x0:x1] = (sub == cv2.GC_FGD) | (sub == cv2.GC_PR_FGD)
    return out & zone


# заливка от точно объекта и точно фона, останавливается на краях
def _watershed(sc, zone, seed, fg, bg, z):
    markers = np.zeros(sc.shape, np.int32)
    markers[~dilate(zone, 2) | bg] = 1
    sure = fg.copy()
    if seed is not None:
        sure |= erode(seed, 1)
    sure &= zone & ~bg
    if not sure.any():
        return zone.copy()
    markers[sure] = 2
    cv2.watershed(np.ascontiguousarray(sc.u8[..., ::-1]), markers)
    return morph(markers == 2, cv2.MORPH_CLOSE, 1) & zone


# маска, нарисованная руками: белое - объект
# собирает маску объекта: по цвету, grabcut, watershed или просто многоугольник,
# потом вырезаем exclude
def object_mask(sc, z):
    zone = polygon_mask(z["polygon"], sc.shape)
    method = z.get("mask_method") or "color"
    seed = (color_match(sc, z["colors"]) & zone) if z.get("colors") else None
    if method == "polygon":
        m = zone.copy()
    elif method == "color":
        m = clean_mask(seed if seed is not None else zone.copy(), z)
    elif method in ("grabcut", "watershed"):
        fg = polys_mask(z.get("fg_hint"), sc.shape)
        bg = polys_mask(z.get("bg_hint"), sc.shape)
        if z.get("bg_colors"):
            bg |= color_match(sc, z["bg_colors"]) & zone
        m = (_grabcut if method == "grabcut" else _watershed)(sc, zone, seed, fg, bg, z)
        m = clean_mask(m, z)
    else:
        raise ValueError(f"unknown mask_method: {method}")
    if z.get("fill_holes"):
        m = binary_fill_holes(m) & zone
    m = m & ~polys_mask(z.get("exclude"), sc.shape)
    return zone, m


# насколько контур маски совпадает с настоящим краем на картинке
def edge_confidence(sc, m, ec):
    lab = cv2.cvtColor(cv2.GaussianBlur(sc.u8, (0, 0), ec["blur"]), cv2.COLOR_RGB2LAB).astype(np.float32)
    mu = m.astype(np.uint8)
    sd = cv2.distanceTransform(mu, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - mu, cv2.DIST_L2, 5)
    sd = cv2.GaussianBlur(sd, (0, 0), 1.5)
    gx, gy = cv2.Sobel(sd, cv2.CV_32F, 1, 0), cv2.Sobel(sd, cv2.CV_32F, 0, 1)
    g = np.hypot(gx, gy) + 1e-6
    gx, gy = gx / g, gy / g
    edge = m & ~erode(m, 1)
    ys, xs = np.nonzero(edge)
    off = ec["offset"]
    xi = (xs + gx[ys, xs] * off).astype(np.float32)[None]
    yi = (ys + gy[ys, xs] * off).astype(np.float32)[None]
    xo = (xs - gx[ys, xs] * (off + 1)).astype(np.float32)[None]
    yo = (ys - gy[ys, xs] * (off + 1)).astype(np.float32)[None]
    ci = cv2.remap(lab, xi, yi, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
    co = cv2.remap(lab, xo, yo, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
    H, W = sc.shape
    # край кадра контуром не считаем
    inner = (ys > 0) & (ys < H - 1) & (xs > 0) & (xs < W - 1)
    return ys[inner], xs[inner], np.linalg.norm(ci - co, axis=1)[inner]


# там где край маски не похож на настоящий, движение глушим, чтоб не рвало фон
def edge_damp(sc, m, ec):
    ys, xs, de = edge_confidence(sc, m, ec)
    pts = np.zeros(sc.shape, np.float32)
    wsum = np.zeros(sc.shape, np.float32)
    pts[ys, xs] = np.clip((de - ec["de_min"]) / ec["de_range"], 0, 1)
    wsum[ys, xs] = 1
    R = ec["radius"]
    gw = gaussian_filter(wsum, R)
    conf = gaussian_filter(pts, R) / (gw + 1e-4)
    near = np.clip(gw * np.pi * R, 0, 1)
    return (1 - near * (1 - conf)).astype(np.float32), (ys, xs, de)


# картинка и всё, что от неё часто нужно эффектам
class Scene:
    def __init__(self, img_u8, cfg):
        self.u8 = img_u8
        self.img = img_u8.astype(np.float32)
        self.H, self.W = img_u8.shape[:2]
        self.shape = (self.H, self.W)
        self.hsv = rgb_to_hsv(img_u8)
        self.lum = self.img.mean(-1)
        self.yy, self.xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        self.N = int(cfg["frames"])
        self.seed = int(cfg["seed"])

    def rng(self, effect, index):
        return np.random.default_rng([self.seed, _RNG_STREAM[effect], index])


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / max(float(np.hypot(*v)), 1e-6)


# средний цвет вокруг внутри маски, от него считаются блики
def _local_mean(img, m, sigma):
    wf = m.astype(np.float32)
    norm = gaussian_filter(wf, sigma) + 1e-3
    local = np.stack([gaussian_filter(img[..., c] * wf, sigma) / norm for c in range(3)], -1)
    return np.where(m[..., None], local, img)


# путь по воде от истока в обход берегов
def _geodesic(mask, starts, metric):
    costs = np.where(mask, 1.0, np.inf)
    if metric == "euclidean":
        mcp = MCP_Geometric(costs)
    else:  # chessboard: шаг в любую из 8 сторон стоит одинаково
        mcp = MCP(costs, fully_connected=True)
    d, _ = mcp.find_costs(starts)
    return d


# для каждого пикселя воды: сколько плыть от истока. по этому числу и бегут полосы
def _flow_distance(water, source, metric):
    src = root_mask(source, water.shape)
    seeds = src & water
    sy, sx = np.nonzero(src)
    if not seeds.any():  # исток чуть мимо воды, берём ближайшую точку воды
        ys, xs = np.nonzero(water)
        d2 = (xs[:, None] - sx[None]) ** 2 + (ys[:, None] - sy[None]) ** 2 \
            if len(sx) * len(xs) < 5e7 else (xs - sx.mean()) ** 2 + (ys - sy.mean()) ** 2
        k = np.unravel_index(np.argmin(d2), d2.shape)[0]
        seeds = np.zeros_like(water)
        seeds[ys[k], xs[k]] = True
    starts = list(zip(*np.nonzero(seeds)))
    dist = _geodesic(water, starts, metric)
    cy0, cx0 = np.mean(np.nonzero(seeds), axis=1)
    lab, n = label(water & ~np.isfinite(dist))
    for c in range(1, n + 1):
        comp = lab == c
        cy, cx = np.nonzero(comp)
        k = np.argmin((cx - cx0) ** 2 + (cy - cy0) ** 2)
        d = _geodesic(comp, [(cy[k], cx[k])], metric)
        dist[comp] = d[comp] + math.hypot(cx[k] - cx0, cy[k] - cy0)
    return np.where(water, dist, 0).astype(np.float32), seeds


# готовит воду: маска, пена, путь от истока, неровность полос
def water_setup(sc, z, idx):
    if (z.get("mask_method") or "color") == "color":
        zone = polygon_mask(z["polygon"], sc.shape)
        col = color_match(sc, z["colors"]) & zone
        col = morph(col, cv2.MORPH_OPEN, z["open_iter"])
        foam = (color_match(sc, z["foam_colors"]) & zone) if z.get("foam_colors") \
            else np.zeros(sc.shape, bool)
        water = col | (foam & dilate(col, z["foam_reach"]))
        water = morph(water, cv2.MORPH_CLOSE, z["close_iter"]) & zone
        water = remove_small(water, z["min_area"])
        water = water & ~polys_mask(z.get("exclude"), sc.shape)
    else:
        zone, water = object_mask(sc, z)
        foam = (color_match(sc, z["foam_colors"]) & water) if z.get("foam_colors") \
            else np.zeros(sc.shape, bool)
    if not water.any():
        return None
    metric = z["flow_metric"]
    if metric == "direction":
        # прямой поток: волны идут в одну сторону
        fx, fy = z["flow_direction"]
        proj = (sc.xx * fx + sc.yy * fy) / max(math.hypot(fx, fy), 1e-6)
        s = np.where(water, proj - proj[water].min(), 0).astype(np.float32)
        seeds = water & (s < 1)
    elif metric == "ellipse":
        # кольца от точки, сплюснутые перспективой
        sx, sy = _pts(z["source"])[0]
        s = np.sqrt((sc.xx - sx) ** 2 + ((sc.yy - sy) / z["ellipse_aspect"]) ** 2)
        s = np.where(water, s, 0).astype(np.float32)
        seeds = water & (s < 2)
    else:
        s, seeds = _flow_distance(water, z["source"], metric)

    img = sc.img
    wf = water.astype(np.float32)
    local = _local_mean(img, water, z["local_sigma"])
    contrast = img.mean(-1) - local.mean(-1)
    t0, t1 = z["highlight"]
    hl = gaussian_filter(np.clip((contrast - t0) / t1, 0, 1) * wf, z["highlight_blur"])
    foam_m = gaussian_filter((foam & water).astype(np.float32), z["foam_blur"])
    hl = hl * (1 - np.clip(foam_m, 0, 1))
    phase = gaussian_filter(sc.rng("water", idx).random(sc.shape).astype(np.float32),
                            z["foam_noise_sigma"])
    phase = (phase - phase.min()) / (phase.max() - phase.min() + 1e-6) * z["foam_phase_spread"]
    w0, w1 = z["wavelength"]
    lam = w0 + (w1 - w0) * (sc.yy / sc.H)
    if z["smooth_flow"] > 0:  # сглаживаем, иначе полосы ломаются ёлочкой
        sig = z["smooth_flow"]
        s = np.where(water, gaussian_filter(s * wf, sig) / (gaussian_filter(wf, sig) + 1e-4), 0)
        s = s.astype(np.float32)
    irr_c, irr_f = z["irregularity"]
    sc_c, sc_f = z["irregularity_scale"]
    wrng = np.random.default_rng([sc.seed, _RNG_STREAM["water"], idx, 1])
    psi = np.zeros(sc.shape, np.float32)
    if irr_c or irr_f:  # фронт волны чуть кривой, чтоб не было ровных полос
        psi = irr_c * _smooth_noise(wrng, sc.shape, sc_c) + irr_f * _smooth_noise(wrng, sc.shape, sc_f)
    if z["wave_jitter"]:  # ширина полос плавает пятнами
        lam = lam * (1 + z["wave_jitter"] * _smooth_noise(wrng, sc.shape, 60))
    bamp = None
    if z["patchiness"]:  # где-то блики ярче, где-то почти пропадают
        prng = np.random.default_rng([sc.seed, _RNG_STREAM["water"], idx, 2])
        bamp = np.clip(1 - z["patchiness"] * (0.5 + 0.5 * _smooth_noise(prng, sc.shape, z["patch_scale"])),
                       0, 1).astype(np.float32)[..., None]
    w = dict(kind="water", label=z["label"], params=z, zone=zone, allowed=water, psi=psi, bamp=bamp,
             mask=water, foam=foam & water, s=s, seeds=seeds, local=local,
             hl=hl[..., None], foam_m=foam_m[..., None], foam_phase=phase, lam=lam)
    if z["mode"] == "flow":
        _water_flow_setup(sc, w, z, idx)
    return w


def _cnoise(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap") \
        + 1j * gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return (a / (np.abs(a).std() + 1e-9)).astype(np.complex64)


# второй режим воды: мазки едут вдоль течения
def _water_flow_setup(sc, w, z, idx):
    m = w["mask"]
    mf = m.astype(np.float32)
    rng = np.random.default_rng([sc.seed, _RNG_STREAM["water"], idx, 9])
    sg = z["flow_smooth"]
    s = gaussian_filter(w["s"] * mf, sg) / (gaussian_filter(mf, sg) + 1e-4)
    gy, gx = np.gradient(s)
    n = np.hypot(gx, gy) + 1e-6
    gx, gy = gx / n, gy / n
    lsm = gaussian_filter(sc.lum, 1.2)
    iy, ix = np.gradient(lsm)
    Jxx, Jyy, Jxy = (gaussian_filter(a, sg) for a in (ix * ix, iy * iy, ix * iy))
    th = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + np.pi / 2
    coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-3)
    tx, ty = np.cos(th), np.sin(th)
    sgn = np.sign(tx * gx + ty * gy)
    sgn[sgn == 0] = 1
    k = z["stroke_follow"] * coh * np.abs(tx * gx + ty * gy)
    vx = gaussian_filter((1 - k) * gx + k * sgn * tx, sg / 2)
    vy = gaussian_filter((1 - k) * gy + k * sgn * ty, sg / 2)
    n = np.hypot(vx, vy) + 1e-6
    fall = np.clip(distance_transform_edt(m) / max(z["advect_edge"], 1), 0, 1).astype(np.float32)
    fall = fall * fall * (3 - 2 * fall)
    align = np.clip(gaussian_filter(coh * np.abs(tx * gx + ty * gy), sg) * 1.6, 0, 1)
    fall = (fall * (z["cross_speed"] + (1 - z["cross_speed"]) * align)).astype(np.float32)
    w["vx"], w["vy"], w["fall"] = (vx / n).astype(np.float32), (vy / n).astype(np.float32), fall
    w["uoff"] = (z["flow_phase_noise"] * _smooth_noise(rng, sc.shape, z["flow_phase_scale"])).astype(np.float32)
    if z["ripple"]:
        w["rip"] = (_cnoise(rng, sc.shape, z["ripple_scale"] / 2.5), _cnoise(rng, sc.shape, z["ripple_scale"] / 2.5))
    C = np.zeros(sc.shape, np.complex64)
    for q in z["shimmer_waves"]:
        C += _cnoise(rng, sc.shape, z["shimmer_patch"]) * np.exp(2j * np.pi * s / (w["lam"] * q))
    w["C"] = C / (np.abs(C).max() * 0.6 + 1e-6)


def _water_flow_frame(sc, base, w, i):
    z = w["params"]
    N = sc.N
    out = base
    if z["advect"]:
        acc = np.zeros_like(base)
        for off in (0.0, 0.5):
            u = (i / N + w["uoff"] + off) % 1.0
            dsp = (u - 0.5) * z["advect"] * w["fall"]
            a = cv2.remap(base, (sc.xx - dsp * w["vx"]).astype(np.float32),
                          (sc.yy - dsp * w["vy"]).astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
            acc += a * (1 - np.abs(2 * u - 1))[..., None]
        if z["sharpen"]:
            acc = acc + z["sharpen"] * w["fall"][..., None] * (acc - cv2.GaussianBlur(acc, (0, 0), 1.5))
        out = acc
    if z["ripple"]:
        e = np.exp(-2j * np.pi * i / N)
        dx = z["ripple"] * w["fall"] * (w["rip"][0] * e).real
        dy = z["ripple"] * w["fall"] * (w["rip"][1] * e).real
        out = cv2.remap(out, (sc.xx - dx).astype(np.float32), (sc.yy - dy).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if z["shimmer"]:
        b = np.clip((w["C"] * np.exp(-2j * np.pi * z["cycles"] * i / N)).real, -1, 1)
        gain = 1 + z["shimmer"] * b
        out = w["local"] + (out - w["local"]) * (1 + w["hl"] * (gain[..., None] - 1))
    return np.where(w["mask"][..., None], np.clip(out, 0, 255), base)


# кадр воды: полоски света едут от истока по течению
def water_frame(sc, base, w, i):
    z = w["params"]
    if "fall" in w:
        return _water_flow_frame(sc, base, w, i)
    p = z["cycles"] * i / sc.N
    L = z["gain_levels"]
    band = 0.5 + 0.5 * np.cos(2 * np.pi * (w["s"] / w["lam"] + w["psi"] - p))
    if L:
        band = np.round(band * L) / L
    gain = (z["gain_min"] + (z["gain_max"] - z["gain_min"]) * band)[..., None]
    if w.get("bamp") is not None:
        gain = 1 + w["bamp"] * (gain - 1)
    out = w["local"] + (base - w["local"]) * (1 + w["hl"] * (gain - 1))
    FL = z["foam_levels"]
    churn = 0.5 + 0.5 * np.sin(2 * np.pi * (p + w["foam_phase"]) + w["s"] / z["foam_flow_scale"])
    churn = np.round(churn * FL) / FL
    out = out + w["foam_m"] * (churn[..., None] - 0.5) * z["foam_strength"]
    return np.where(w["mask"][..., None], np.clip(out, 0, 255), base)


# ветер траву не двигает, а гоняет по ней пятна света
def wind_setup(sc, z, idx):
    zone, m = object_mask(sc, z)
    if not m.any():
        return None
    # у краёв зоны и у вырезанных кусков гасим плавно, без швов
    zone_eff = zone & ~polys_mask(z.get("exclude"), sc.shape)
    w = edge_falloff(zone_eff, z["edge_feather"]) * gaussian_filter(m.astype(np.float32), 1.0)
    d = _unit(z["direction"])
    proj = sc.xx * d[0] + sc.yy * d[1]
    s = (proj - proj[m].min()).astype(np.float32)
    local = _local_mean(sc.img, m, z["local_sigma"])
    t0, t1 = z["highlight"]
    hl = np.clip((sc.lum - local.mean(-1) - t0) / t1, 0, 1) * m
    w0, w1 = z["wavelength"]
    lam = w0 + (w1 - w0) * (sc.yy / sc.H)
    # порывы неровные: фронт кривой, сила пятнами
    rng = sc.rng("wind", idx)
    psi = np.pi * z["irregularity"] * _smooth_noise(rng, sc.shape, z["patch_scale"])
    amp = np.clip(1 - z["patchiness"] * (0.5 + 0.5 * _smooth_noise(rng, sc.shape, z["patch_scale"])),
                  0, 1)
    out = dict(kind="wind", label=z["label"], params=z, zone=zone, mask=m, allowed=w > 0,
               w=w[..., None], s=s, local=local, hl=hl[..., None], lam=lam, d=d,
               psi=psi, amp=amp)
    if z["mode"] == "gust":
        # редкие мягкие пятна, как тени от облаков
        grng = np.random.default_rng([sc.seed, _RNG_STREAM["wind"], idx, 7])
        g = flow_noise(grng, sc.shape, z["scale"], sc.N, z["direction"], z["directionality"],
                       z["aniso"], harmonics=z["harmonics"])
        if z["detail"]:
            g = g + z["detail"] * flow_noise(grng, sc.shape, z["detail_scale"], sc.N, z["direction"],
                                             z["directionality"], z["aniso"])
        g /= g.std()
        g = _perspective(sc, g, m, z["perspective"])
        g = np.maximum(g - z["sparsity"], 0)
        g -= g.mean(0, keepdims=True)
        g /= np.percentile(np.abs(g), 99.5) + 1e-6
        out["g"] = np.clip(g, -1, 1).astype(np.float32)
    return out


def wind_frame(sc, base, W, i):
    z = W["params"]
    if "g" in W:
        b2 = W["g"][i][..., None]
    else:
        b2 = (W["amp"] * np.cos(2 * np.pi * (W["s"] / W["lam"] - z["cycles"] * i / sc.N)
                                + W["psi"]))[..., None]
    out = W["local"] + (base - W["local"]) * (1 + W["hl"] * z["contrast"] * b2)
    out = out * (1 + z["brightness"] * b2)
    return np.clip(base + (out - base) * W["w"], 0, 255)


# гладкий шум от -1 до 1
def _smooth_noise(rng, shape, sigma):
    n = gaussian_filter(rng.random(shape).astype(np.float32), sigma, mode="wrap")
    n -= n.mean()
    return n / (np.abs(n).max() + 1e-6)


# гладкий шум, вытянутый в одну сторону (для лучей)
def _aniso_noise(rng, sc, d, s_along, s_across):
    H, W = sc.shape
    D = int(math.ceil(math.hypot(H, W))) + 4
    n = gaussian_filter(rng.random((D, D)).astype(np.float32), (s_along, s_across), mode="wrap")
    n -= n.mean()
    n /= np.abs(n).max() + 1e-6
    perp = (-d[1], d[0])
    along = (sc.xx - W / 2) * d[0] + (sc.yy - H / 2) * d[1] + D / 2
    across = (sc.xx - W / 2) * perp[0] + (sc.yy - H / 2) * perp[1] + D / 2
    return cv2.remap(n, across.astype(np.float32), along.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# шум, который едет и через N кадров приходит обратно, чтоб анимация зацикливалась
def loop_noise(rng, shape, sigma, drift, N, jitter=0.0):
    H, W = shape
    spec = np.fft.fft2(rng.standard_normal(shape))
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    spec *= np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    turns = np.rint((fx * drift[0] + fy * drift[1]) * N + jitter * rng.standard_normal(shape))
    out = np.stack([np.fft.ifft2(spec * np.exp(-2j * np.pi * turns * i / N)).real for i in range(N)])
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return out.astype(np.float32)


# тоже зацикленный шум, но без повторяющегося узора
def flow_noise(rng, shape, scale, N, direction=None, directionality=1.0, aniso=1.0,
               bandwidth=0.6, harmonics=(1.0,)):
    H, W = shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    d = _unit(direction) if direction is not None else np.array([1.0, 0.0], np.float32)
    fa = fx * d[0] + fy * d[1]
    fc = -fx * d[1] + fy * d[0]
    out = np.zeros((N,) + shape, np.float32)
    for h, a in enumerate(harmonics, 1):
        if not a:
            continue
        f0 = h ** 0.5 / scale
        fr = np.sqrt(fa ** 2 + (fc * aniso) ** 2)
        spec = np.exp(-0.5 * ((fr - f0) / (bandwidth * f0)) ** 2)
        if direction is not None and directionality:
            spec = spec * (1 + directionality * np.tanh(fa / (0.35 * f0))) / 2
        zc = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        C = np.fft.ifft2(np.fft.fft2(zc) * spec)
        C /= np.sqrt((np.abs(C) ** 2).mean() / 2) + 1e-12
        for i in range(N):
            out[i] += a * (C * np.exp(-2j * np.pi * h * i / N)).real
    return out / (out.std() + 1e-9)


# вдали пятна мельче, вблизи крупнее
def _perspective(sc, frames, mask, k01):
    if not k01 or (k01[0] == 1 and k01[1] == 1):
        return frames
    ys = np.nonzero(mask.any(1))[0]
    y0, y1 = ys[0], ys[-1]
    k = k01[0] + (k01[1] - k01[0]) * np.clip((np.arange(sc.H) - y0) / max(y1 - y0, 1), 0, 1)
    yy = np.cumsum(1 / k).astype(np.float32)
    mx = (sc.W / 2 + (sc.xx - sc.W / 2) / k[:, None]).astype(np.float32)
    my = np.repeat(yy[:, None], sc.W, 1)
    return np.stack([cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) for f in frames])


# пятна света: мягкие, точками или нитями как от воды
def _light_field(sc, rng, z, w):
    m = flow_noise(rng, sc.shape, z["scale"], sc.N, z.get("direction"), z["directionality"],
                   z["aniso"], harmonics=z["harmonics"])
    if z.get("detail"):
        m += z["detail"] * flow_noise(rng, sc.shape, z["detail_scale"], sc.N, z.get("direction"),
                                      z["directionality"], z["aniso"])
        m /= m.std()
    pat = z.get("pattern", "soft")
    if pat == "spots":
        m = np.maximum(m - z["sparsity"], 0)
    elif pat == "caustic":  # яркие нити как от воды
        m = np.clip(1 - np.abs(m) * z["caustic_sharp"], 0, 1) ** 2
        cb = z["caustic_blur"]
        m = gaussian_filter(m, (0, cb, cb))
    if z.get("flicker"):
        t = 2 * np.pi * np.arange(sc.N) / sc.N
        ph = rng.uniform(0, 2 * np.pi, 2)
        fl = np.cos(t + ph[0]) + 0.5 * np.cos(2 * t + ph[1])
        m = m / (np.abs(m).max() + 1e-6) + z["flicker"] * fl[:, None, None]
    m -= m.mean(0, keepdims=True)
    m /= np.percentile(np.abs(m), 99.5) + 1e-6
    m = np.clip(m, -1, 1)
    if z.get("one_sided", True):
        # свет только подсвечивает, темнее чем нарисовано не бывает
        m = ((m + 1) / 2) ** z.get("one_sided_gamma", 2.0)
    return (m * w).astype(np.float32)


# свет гуляет по светлому, тени не трогаем
def _light_weight(sc, z):
    if not z.get("light"):
        return 1.0
    lo, hi = z["light"]
    return np.clip((gaussian_filter(sc.lum, 1.5) - lo) / max(hi - lo, 1), 0, 1) ** z["light_gamma"]


# игра света по камням и стенам
def light_setup(sc, z, idx):
    zone = polygon_mask(z["polygon"], sc.shape) if z.get("polygon") else np.ones(sc.shape, bool)
    zone &= ~polys_mask(z.get("exclude"), sc.shape)
    if not zone.any():
        return None
    w = edge_falloff(zone, z["edge_feather"]) if not zone.all() else np.ones(sc.shape, np.float32)
    if z.get("colors"):
        w = w * gaussian_filter(color_match(sc, z["colors"]).astype(np.float32), 3.0)
    if z.get("near_water"):  # отблески только рядом с водой
        wm = getattr(sc, "water_mask", None)
        if wm is not None and wm.any():
            dw = distance_transform_edt(~wm)
            w = w * (np.clip(1 - dw / z["near_water"], 0, 1) ** 1.5 * (~wm))
        else:
            w = w * 0
    w = (w * _light_weight(sc, z)).astype(np.float32)
    if not (w > 0.02).any():
        return None
    m = _light_field(sc, sc.rng("light", idx), z, w)
    return dict(kind="light", label=z["label"], params=z, zone=zone, mask=w > 0.02, allowed=w > 0,
                w=w, m=m, tint=np.asarray(z["tint"], np.float32))


def light_frame(sc, base, E, i):
    z = E["params"]
    m = E["m"][i][..., None] * E["tint"]
    return np.clip(base * (1 + z["strength"] * m) + z["lift"] * m, 0, 255)


# облака не двигаем, только по ним гуляет свет
def clouds_setup(sc, z, idx):
    zone, m = object_mask(sc, z)
    if not m.any():
        return None
    w = edge_falloff(m, z["edge_feather"])
    if z["mode"] == "light":
        lw = (w * _light_weight(sc, z)).astype(np.float32)
        lm = _light_field(sc, sc.rng("clouds", idx), z, lw)
        return dict(kind="clouds", label=z["label"], params=z, zone=zone, mask=m, allowed=lw > 0,
                    w=lw, lm=lm, tint=np.asarray(z["tint"], np.float32))
    bf = float(z["border_feather"])
    if bf > 0:  # у краёв кадра тоже гасим, там бывают полосы от jpeg
        H, W = sc.shape
        db = np.minimum(np.minimum(sc.xx, W - 1 - sc.xx), np.minimum(sc.yy, H - 1 - sc.yy))
        t = np.clip(db / bf, 0, 1)
        w = w * (t * t * (3 - 2 * t))
    rng = sc.rng("clouds", idx)
    n = [_smooth_noise(rng, sc.shape, z["scale"]) for _ in range(4)]
    return dict(kind="clouds", label=z["label"], params=z, zone=zone, mask=m, allowed=w > 0,
                w=w, n=n)


def clouds_frame(sc, base, C, i):
    z = C["params"]
    if "lm" in C:
        m = C["lm"][i][..., None] * C["tint"]
        return np.clip(base * (1 + z["strength"] * m) + z["lift"] * m, 0, 255)
    t = 2 * np.pi * i / sc.N
    A, (sx, sy) = z["amplitude"], z["sway"]
    n1, n2, n3, n4 = C["n"]
    dx = C["w"] * (A * (n1 * np.cos(t) + n2 * np.sin(t)) + sx * np.sin(t))
    dy = C["w"] * (A * (n3 * np.cos(t) + n4 * np.sin(t)) + sy * np.sin(t))
    return cv2.remap(base, (sc.xx - dx).astype(np.float32), (sc.yy - dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# лучи переливаются, каждый в своём ритме
def rays_setup(sc, z, idx):
    zone = polygon_mask(z["polygon"], sc.shape)
    if not zone.any():
        return None
    lo, hi = z["light"]
    w = edge_falloff(zone, z["edge_feather"]) * np.clip((sc.lum - lo) / max(hi - lo, 1), 0, 1)
    if z.get("colors"):
        w = w * gaussian_filter(color_match(sc, z["colors"]).astype(np.float32), 2.0)
    d = _unit(z["direction"])
    rng = sc.rng("rays", idx)
    L, Wd = z["shaft_length"], z["shaft_width"]
    psi = np.pi * z["flicker_spread"] * _aniso_noise(rng, sc, d, L, Wd)
    amp = 0.5 + 0.5 * _aniso_noise(rng, sc, d, L, Wd)
    return dict(kind="rays", label=z["label"], params=z, zone=zone, mask=w > 0.02, allowed=w > 0,
                w=(w * amp).astype(np.float32), psi=psi, d=d)


def rays_frame(sc, base, R, i):
    z = R["params"]
    f = np.cos(2 * np.pi * z["cycles"] * i / sc.N + R["psi"])
    if z.get("one_sided", True):  # лучи только вспыхивают, темнее не становятся
        f = (0.5 + 0.5 * f) ** 2
    m = z["strength"] * f * R["w"]
    return np.clip(base * (1 + m[..., None]), 0, 255)


# всё что светится: кристаллы, огоньки, лианы. у каждого объекта своя фаза
def glow_setup(sc, z, idx):
    zone, m = object_mask(sc, z)
    if not m.any():
        return None
    lab, n = label(m)
    rng = sc.rng("glow", idx)
    # phase_spread меньше 1: мигают почти вместе
    ph = (rng.uniform(0, 2 * np.pi, n + 1) * z.get("phase_spread", 1.0)).astype(np.float32)
    c0, c1 = z["cycles"]
    cyc = rng.integers(int(c0), int(c1) + 1, n + 1)
    amp = rng.uniform(*z["amp_jitter"], n + 1).astype(np.float32)
    hs = float(z["halo_sigma"])
    region = dilate_round(m, int(math.ceil(3 * hs)) + 2)
    pa, pp, pc = amp[lab] * m, ph[lab], cyc[lab]
    fields = []
    for c in np.unique(cyc[1:]):
        mc = (pc == c) & m
        ac, as_ = pa * mc * np.cos(pp), pa * mc * np.sin(pp)
        fields.append((int(c),
                       np.stack([gaussian_filter(sc.img[..., k] * ac, hs) for k in range(3)], -1),
                       np.stack([gaussian_filter(sc.img[..., k] * as_, hs) for k in range(3)], -1),
                       gaussian_filter(ac, z["core_blur"]), gaussian_filter(as_, z["core_blur"])))
    out = dict(kind="glow", label=z["label"], params=z, zone=zone, mask=m, allowed=region,
               region=region, fields=fields, count=n)
    if z.get("pattern", "pulse") == "glint":
        out["glint"] = _glint_setup(sc, z, lab, n, m, rng)
    return out


# чтоб кристаллы не мигали, а по ним ехал блик
def _glint_setup(sc, z, lab, n, m, rng):
    idx = np.arange(1, n + 1)
    ys, xs = sc.yy[m], sc.xx[m]
    L = lab[m]
    cnt = np.bincount(L, minlength=n + 1).astype(np.float64) + 1e-9
    mx, my = np.bincount(L, xs, n + 1) / cnt, np.bincount(L, ys, n + 1) / cnt
    dx, dy = xs - mx[L], ys - my[L]
    cxx = np.bincount(L, dx * dx, n + 1) / cnt
    cyy = np.bincount(L, dy * dy, n + 1) / cnt
    cxy = np.bincount(L, dx * dy, n + 1) / cnt
    th = 0.5 * np.arctan2(2 * cxy, cxx - cyy)  # главная ось объекта
    th = th + np.deg2rad(z["glint_dir_jitter"]) * rng.uniform(-1, 1, n + 1)
    th = th + np.pi * rng.integers(0, 2, n + 1)  # в какую сторону едет, случайно
    pr = dx * np.cos(th[L]) + dy * np.sin(th[L])
    lo = np.full(n + 1, np.inf); hi = np.full(n + 1, -np.inf)
    np.minimum.at(lo, L, pr); np.maximum.at(hi, L, pr)
    u = (pr - lo[L]) / np.maximum(hi[L] - lo[L], 1.0)  # от 0 до 1 вдоль объекта
    phi = rng.uniform(0, 2 * np.pi, n + 1)
    U = np.zeros(sc.shape, np.float32); PH = np.zeros(sc.shape, np.float32)
    if z.get("glint_period_px"):
        # на длинных несколько коротких бликов, иначе блик проносится слишком быстро
        U[m] = pr / float(z["glint_period_px"])
    else:
        U[m] = u / z["glint_period"]
    PH[m] = phi[L]
    if z.get("glint_irregularity"):
        nz = gaussian_filter(rng.standard_normal(sc.shape).astype(np.float32), z["glint_irregularity_scale"])
        nz /= nz[m].std() + 1e-6
        PH[m] += 2 * np.pi * float(z["glint_irregularity"]) * nz[m]
    return dict(U=U, PH=PH)


def _glint_frame(sc, base, g, i):
    z, G = g["params"], g["glint"]
    w = ((1 + np.cos(2 * np.pi * (G["U"] - i / sc.N) + G["PH"])) / 2) ** z["glint_power"]
    w = (w * g["mask"]).astype(np.float32)
    core = gaussian_filter(w, z["core_blur"])
    halo = np.stack([gaussian_filter(sc.img[..., k] * w, z["halo_sigma"]) for k in range(3)], -1)
    out = base * (1 + z["amplitude"] * core[..., None]) + z["halo_strength"] * halo
    return np.where(g["region"][..., None], np.clip(out, 0, 255), base)


def glow_frame(sc, base, g, i):
    z = g["params"]
    if g.get("glint") is not None:
        return _glint_frame(sc, base, g, i)
    core = np.zeros(sc.shape, np.float32)
    halo = np.zeros(sc.shape + (3,), np.float32)
    for c, HC, HS, CC, CS in g["fields"]:
        t = 2 * np.pi * c * i / sc.N
        st, ct = np.float32(np.sin(t)), np.float32(np.cos(t))
        core += CC * st + CS * ct
        halo += HC * st + HS * ct
    out = base * (1 + z["amplitude"] * core[..., None]) + z["halo_strength"] * halo
    return np.where(g["region"][..., None], np.clip(out, 0, 255), base)


# растение качается: либо по маске с закраской фона под листьями, либо изгибом зоны
def plant_setup(sc, z, idx):
    mode = z["mode"]
    soft = None
    if mode == "warp" and z.get("warp_mask"):
        # гнём только внутри силуэта, фон вокруг стоит
        zone, m = object_mask(sc, z)
        if m.any():
            margin = max(int(z["warp_margin"]),
                         math.ceil(z["amplitude"] * (1 + z["along_ratio"])) + 1)
            soft = gaussian_filter(dilate_round(m, margin).astype(np.float32), z["warp_soft"])
    elif mode == "warp":
        zone = polygon_mask(z["polygon"], sc.shape)
        m = zone.copy()
    else:
        zone, m = object_mask(sc, z)
    if not m.any():
        return None
    amp = float(z["amplitude"])
    reach_px = math.ceil(amp * (1 + z["along_ratio"])) + 2

    rm = root_mask(z["root"], sc.shape)
    for r in z.get("roots") or []:
        rm |= root_mask(r, sc.shape)
    dist = cv2.distanceTransform((~rm).astype(np.uint8), cv2.DIST_L2, 5)
    reach = z.get("reach") or float(dist[m].max())
    w = np.clip(dist / max(reach, 1.0), 0, 1) ** z["power"]
    # у границы зоны качание плавно сходит на нет
    w = w * edge_falloff(zone, z["edge_feather"])
    if soft is not None:
        w = w * soft
    edge_pts = None
    if z.get("edge_check") and (mode != "warp" or z.get("warp_mask")):
        damp, edge_pts = edge_damp(sc, m, z["edge_check"])
        w = w * damp

    if z.get("sway_direction"):
        perp = _unit(z["sway_direction"])
        axis = np.array([perp[1], -perp[0]], np.float32)
    else:
        ry, rx = np.nonzero(rm)
        my_, mx_ = np.nonzero(m)
        axis = _unit([mx_.mean() - rx.mean(), my_.mean() - ry.mean()])
        perp = np.array([-axis[1], axis[0]], np.float32)

    rng = sc.rng("plants", idx)
    ph = rng.uniform(0, 2 * np.pi, 2)
    parts = _plant_parts(sc, z, idx, m, zone, rm, mode, perp) if z.get("parts") else None
    pnoise = None
    if z["phase_noise_amount"]:
        # отдельный рандом, чтоб не сдвинуть уже выбранные фазы
        prng = np.random.default_rng([sc.seed, _RNG_STREAM["plants"], idx, 2])
        pnoise = (np.pi * z["phase_noise_amount"]
                  * _smooth_noise(prng, sc.shape, z["phase_noise_scale"])).astype(np.float32)
    out = dict(kind="plants", mode=mode, label=z["label"], params=z, zone=zone,
               mask=m, root=rm, axis=axis, perp=perp, phase=ph, edge_pts=edge_pts, pnoise=pnoise,
               parts=parts)
    if mode == "warp":
        out["w"] = w
        return out
    support = gaussian_filter(dilate(m, reach_px + 2).astype(np.float32), 2.0)
    if z["alpha_mode"] == "guided":
        r = int(z["alpha_radius"])
        alpha = guided_filter((sc.lum / 255.0).astype(np.float32),
                              dilate(m, z["alpha_dilate"]).astype(np.float32), r, z["alpha_eps"])
        alpha = alpha * dilate(m, z["alpha_dilate"] + r)
    else:
        alpha = gaussian_filter(dilate(m, z["alpha_dilate"]).astype(np.float32), z["alpha_blur"])
    out.update(
        w=w * support,
        # закрашиваем только лист с небольшой каймой, а не всю зону
        hole=dilate(m, z["alpha_dilate"] + z["under_extra"] + z["inpaint_extra"]),
        alpha_src=alpha,
        under_src=gaussian_filter(dilate(m, z["alpha_dilate"] + z["under_extra"])
                                  .astype(np.float32), z["alpha_blur"]))
    return out


# режем растение на кусочки, у каждого своя фаза и сила, чтоб не качалось одним куском
def _plant_parts(sc, z, idx, m, zone, rm, mode, perp):
    q = {**z["parts_defaults"], **(z["parts"] if isinstance(z["parts"], dict) else {})}
    objmode = mode != "warp" or z.get("warp_mask")
    region = dilate(m, 1) if objmode else zone
    n_seg = max(6, int(region.sum() / q["size"] ** 2))
    lab = slic(sc.u8, n_segments=n_seg, compactness=q["compactness"], mask=region, start_label=1,
               channel_axis=-1)
    K = int(lab.max())
    rng = np.random.default_rng([sc.seed, _RNG_STREAM["plants"], idx, 5])
    ph = np.r_[0, q["phase_spread"] * np.pi * rng.uniform(-1, 1, K)]
    am = np.r_[0, rng.uniform(*q["amp_jitter"], K)]
    th = np.r_[0, np.deg2rad(q["dir_jitter"]) * rng.uniform(-1, 1, K)]
    ph2 = np.r_[0, rng.uniform(0, 2 * np.pi, K)]
    rf = region.astype(np.float32)
    sg = max(q["smooth"] * q["size"], 0.8)
    nrm = gaussian_filter(rf, sg) + 1e-4

    def field(vals):
        v = vals[lab] * rf
        return gaussian_filter(v.real.astype(np.float32), sg) / nrm \
            + 1j * gaussian_filter(v.imag.astype(np.float32), sg) / nrm

    S1 = field(am * np.exp(1j * ph)).astype(np.complex64)
    S2 = field(am * np.exp(1j * ph2)).astype(np.complex64)
    R = field(np.exp(1j * th))
    rot = (R / (np.abs(R) + 1e-6)).astype(np.complex64)
    _, (iy, ix) = distance_transform_edt(~rm, return_indices=True)
    tx, ty = -(sc.yy - iy), (sc.xx - ix)
    n = np.hypot(tx, ty) + 1e-3
    tx, ty = gaussian_filter(tx / n, 4), gaussian_filter(ty / n, 4)
    if (tx[m] * perp[0] + ty[m] * perp[1]).mean() < 0:
        tx, ty = -tx, -ty
    a = q["arc"]
    tx, ty = a * tx + (1 - a) * perp[0], a * ty + (1 - a) * perp[1]
    n = np.hypot(tx, ty) + 1e-3
    return dict(S1=S1, S2=S2, rot=rot, tan=((tx + 1j * ty) / n).astype(np.complex64), q=q, count=K)


# все растения вместе: общая маска и закраска фона под ними
def plants_combine(sc, plants):
    if not plants:
        return None
    masked = [p for p in plants if p["mode"] != "warp"]
    P = dict(plants=plants, warp=[p for p in plants if p["mode"] == "warp"], masked=masked)
    if not masked:
        return P
    hole = np.zeros(sc.shape, bool)
    alpha = np.zeros(sc.shape, np.float32)
    under = np.zeros(sc.shape, np.float32)
    reach = np.zeros(sc.shape, np.float32)  # на сколько пикселей максимум уезжает лист в этой точке
    for p in masked:
        hole |= p["hole"]
        alpha = np.maximum(alpha, p["alpha_src"])
        under = np.maximum(under, p["under_src"])
        reach = np.maximum(reach, p["params"]["amplitude"] * (1 + p["params"]["along_ratio"])
                           * gaussian_filter(p["w"], 2.0))
    # у корня лист не двигается, закраска там не нужна
    under = under * np.clip(reach / masked[0]["params"]["under_full_shift"], 0, 1)
    radius = max(p["params"]["inpaint_radius"] for p in masked)
    plate = cv2.inpaint(sc.u8, hole.astype(np.uint8) * 255, radius,
                        cv2.INPAINT_TELEA).astype(np.float32)
    # закраска только там, где был лист
    P.update(hole=hole, alpha=alpha, plate=plate, under=(under * hole)[..., None])
    return P


# на сколько сдвинуть каждый пиксель в этом кадре
def _sway_field(sc, plants, i):
    t = 2 * np.pi * i / sc.N
    dx = np.zeros(sc.shape, np.float32)
    dy = np.zeros(sc.shape, np.float32)
    for p in plants:
        z = p["params"]
        wx, wy = z["wave"]
        if p.get("parts") is not None:  # по кусочкам: у каждого своё движение
            P, q = p["parts"], p["parts"]["q"]
            carrier = t + p["phase"][0] + sc.xx / wx + sc.yy / wy
            glob = np.sin(carrier)
            loc = (P["S1"] * np.exp(1j * carrier)).imag
            fl = (P["S2"] * np.exp(2j * t)).imag
            sw = (1 - q["amount"]) * glob + q["amount"] * loc + q["flutter"] * fl
            along = np.sin(t + p["phase"][1] + sc.xx / wy + sc.yy / wx)
            dvec = P["tan"] * P["rot"]
            k = z["amplitude"] * p.get("gain", 1.0) * p["w"]
            a = z["along_ratio"] * (1 - 0.5 * q["amount"])
            dx += (k * (dvec.real * sw + a * p["axis"][0] * along)).astype(np.float32)
            dy += (k * (dvec.imag * sw + a * p["axis"][1] * along)).astype(np.float32)
            continue
        sway = np.sin(t + p["phase"][0] + sc.xx / wx + sc.yy / wy)
        along = np.sin(t + p["phase"][1] + sc.xx / wy + sc.yy / wx)
        if p.get("pnoise") is not None:  # соседние ветки вразнобой, без бегущей волны
            sway = np.sin(t + p["phase"][0] + sc.xx / wx + sc.yy / wy + p["pnoise"])
            along = np.sin(t + p["phase"][1] + sc.xx / wy + sc.yy / wx + p["pnoise"])
        k = z["amplitude"] * p.get("gain", 1.0) * p["w"]
        a = z["along_ratio"]
        dx += k * (p["perp"][0] * sway + a * p["axis"][0] * along)
        dy += k * (p["perp"][1] * sway + a * p["axis"][1] * along)
    return dx, dy


def plants_frame(sc, base, P, i):
    if P["warp"]:
        dx, dy = _sway_field(sc, P["warp"], i)
        base = cv2.remap(base, sc.xx - dx, sc.yy - dy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    if not P["masked"]:
        return base
    dx, dy = _sway_field(sc, P["masked"], i)
    mx, my = (sc.xx - dx).astype(np.float32), (sc.yy - dy).astype(np.float32)
    # края кадра повторяем, иначе у края просвечивает подложка
    leaves = cv2.remap(base, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.remap(P["alpha"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., None]
    # под листом тоже видно, что делают другие эффекты
    plate = P["plate"] + cv2.GaussianBlur(base - sc.img, (0, 0), 3)
    under = plate * P["under"] + base * (1 - P["under"])
    return under * (1 - a) + leaves * a


# туман из нескольких слоёв разного размера
def mist_setup(sc, z, idx):
    zone = polygon_mask(z["polygon"], sc.shape)
    if not zone.any():
        return None
    d = cv2.distanceTransform(zone.astype(np.uint8), cv2.DIST_L2, 5)
    feather = z.get("feather") or float(d.max())
    dens = np.clip(d / max(feather, 1.0), 0, 1) ** z["falloff"]
    rng = sc.rng("mist", idx)
    noise = gaussian_filter(rng.random(sc.shape).astype(np.float32), z["noise_sigma"], mode="wrap")
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    loop, layers = None, None
    if z["noise_mode"] == "loop":
        loop = loop_noise(rng, sc.shape, z["noise_sigma"], z["drift"], sc.N, z["drift_jitter"])
    elif z["noise_mode"] == "layers":
        # три слоя, вытянутые по ветру. часть стоит, часть плывёт
        lrng = np.random.default_rng([sc.seed, _RNG_STREAM["mist"], idx, 3])
        S = np.zeros(sc.shape, np.float32)
        F = np.zeros((sc.N,) + sc.shape, np.float32)
        for sc_k, wgt in zip(z["layer_scales"], z["layer_weights"]):
            px = z["layer_base_scale"] * sc_k
            S += wgt * flow_noise(lrng, sc.shape, px, 1, z["drift"], 0.0, 1 / z["layer_stretch"])[0]
            F += wgt * flow_noise(lrng, sc.shape, px, sc.N, z["drift"], 0.9, 1 / z["layer_stretch"])
        S /= S.std()
        F /= F.std()
        nn = z["static_share"] * S + (1 - z["static_share"]) * F
        nn = 0.5 + 0.5 * np.tanh(nn / z["layer_softness"])
        layers = (nn * dens * z["opacity"]).astype(np.float32)
    return dict(kind="mist", label=z["label"], params=z, zone=zone, allowed=dens > 0,
                mask=zone, dens=dens, noise=noise, loop=loop, layers=layers,
                color=np.asarray(z["color"], np.float32))


def mist_frame(sc, base, m, i):
    z = m["params"]
    if m.get("layers") is not None:
        a = m["layers"][i][..., None]
        return base * (1 - a) + m["color"] * a
    ddx, ddy = z["drift"]
    N = sc.N

    def shifted(k):
        return np.roll(m["noise"], (int(round(k * ddy)), int(round(k * ddx))), (0, 1))

    if m["loop"] is not None:
        n = m["loop"][i]
    else:
        k = i / N
        n = (1 - k) * shifted(i) + k * shifted(i - N)  # склейка конца с началом, чтоб цикл был без рывка
    a = (np.clip((n - z["threshold"]) * z["contrast"], 0, 1) * m["dens"] * z["opacity"])[..., None]
    return base * (1 - a) + m["color"] * a


# крылья бабочки сжимаются к телу и обратно
def flap_setup(sc, z, idx):
    zone, m = object_mask(sc, z)
    if not m.any():
        return None
    a1, a2 = _pts(z["axis"])[:2].astype(np.float32)
    a = _unit(a2 - a1)
    n = np.array([-a[1], a[0]], np.float32)
    d = (sc.xx - a1[0]) * n[0] + (sc.yy - a1[1]) * n[1]
    hole = dilate(m, 3)
    plate = cv2.inpaint(sc.u8, hole.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
    alpha = gaussian_filter(dilate(m, 1).astype(np.float32), 0.8)
    under = (gaussian_filter(dilate(m, 2).astype(np.float32), 0.8) * hole)[..., None]
    ph = float(sc.rng("flap", idx).uniform(0, 2 * np.pi))
    return dict(kind="flap", label=z["label"], params=z, zone=zone, mask=m,
                allowed=zone | hole | (alpha > 0), d=d, n=n, phase=ph,
                plate=plate, alpha=alpha, under=under)


def flap_frame(sc, base, F, i):
    z = F["params"]
    s = 1 - z["amount"] * F.get("gain", 1.0) * (0.5 - 0.5 * np.cos(2 * np.pi * z["cycles"] * i / sc.N + F["phase"]))
    k = (1 / s - 1) * F["d"]
    mx = (sc.xx + F["n"][0] * k).astype(np.float32)
    my = (sc.yy + F["n"][1] * k).astype(np.float32)
    wing = cv2.remap(base, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.remap(F["alpha"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., None]
    plate = F["plate"] + cv2.GaussianBlur(base - sc.img, (0, 0), 3)
    under = plate * F["under"] + base * (1 - F["under"])
    return under * (1 - a) + wing * a


# пыль в луче, пузыри, споры. каждая живёт пару кадров
def particles_setup(sc, z, idx):
    zone = polygon_mask(z["polygon"], sc.shape)
    ys, xs = np.nonzero(zone)
    if len(xs) == 0:
        return None
    rng = sc.rng("particles", idx)
    l0, l1 = z["life"]
    parts = []
    for _ in range(int(z["count"])):
        k = rng.integers(len(xs))
        parts.append(dict(
            x=float(xs[k]), y=float(ys[k]), start=int(rng.integers(sc.N)),
            life=int(min(sc.N - 1, rng.integers(int(l0), int(l1) + 1))),
            speed=float(rng.uniform(*z["speed"])), size=float(rng.uniform(*z["size"])),
            bright=float(rng.uniform(*z["brightness"])), ph=float(rng.uniform(0, 2 * np.pi))))
    halo = 3.0 if z["style"] == "glow" else 1.0
    reach = math.ceil(max(z["speed"]) * sc.N + 4 * halo * max(z["size"]) + z["wobble"]) + 2
    return dict(kind="particles", label=z["label"], params=z, zone=zone, mask=zone,
                allowed=dilate(zone, reach), parts=parts, d=_unit(z["direction"]),
                color=np.asarray(z["color"], np.float32))


def particles_frame(sc, base, P, i):
    z = P["params"]
    out = base.copy()
    d = P["d"]
    perp = np.array([-d[1], d[0]], np.float32)
    style = z["style"]
    for p in P["parts"]:
        j = (i - p["start"]) % sc.N
        if j >= p["life"]:
            continue
        wob = z["wobble"] * math.sin(2 * math.pi * j / max(p["life"], 1) + p["ph"])
        x = p["x"] + d[0] * p["speed"] * j + perp[0] * wob
        y = p["y"] + d[1] * p["speed"] * j + perp[1] * wob
        a = math.sin(math.pi * (j + 0.5) / p["life"]) * p["bright"]
        s = p["size"]
        R = math.ceil(4 * s * (3 if style == "glow" else 1.5)) + 1
        x0, x1 = max(0, int(x) - R), min(sc.W, int(x) + R + 1)
        y0, y1 = max(0, int(y) - R), min(sc.H, int(y) + R + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        X, Y = sc.xx[y0:y1, x0:x1] - x, sc.yy[y0:y1, x0:x1] - y
        d2 = X * X + Y * Y
        if style == "bubble":
            r = np.sqrt(d2)
            g = 0.8 * np.exp(-((r - s) ** 2) / (2 * (0.35 * s) ** 2)) \
                + np.exp(-((X + 0.35 * s) ** 2 + (Y + 0.35 * s) ** 2) / (2 * (0.3 * s) ** 2))
        elif style == "glow":
            g = np.exp(-d2 / (2 * s * s)) + z["glow_strength"] * np.exp(-d2 / (2 * (3 * s) ** 2))
        else:  # пылинка
            g = np.exp(-d2 / (2 * s * s))
        g = np.clip(a * g, 0, 1)[..., None]
        win = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = win + (P["color"] - win) * g
    return out


# светлячки плавно разгораются и летают по маленьким кругам
def fireflies_setup(sc, z, idx):
    zone = polygon_mask(z["polygon"], sc.shape)
    ys, xs = np.nonzero(zone)
    if len(xs) == 0:
        return None
    rng = sc.rng("fireflies", idx)
    if z["motion"] == "smooth":
        # плавно: без выпадающих кадров, летают по маленькому кругу, штук поменьше
        n = max(1, int(round(z["count"] * z["count_scale"])))
        k = rng.integers(len(xs), size=n)
        flies = [dict(x=float(xs[k[j]]), y=float(ys[k[j]]), r=float(rng.uniform(*z["orbit"])),
                      ph=float(rng.uniform(0, 2 * np.pi)), bph=float(rng.uniform(0, 2 * np.pi)),
                      sgn=int(rng.choice([-1, 1]))) for j in range(n)]
        reach = math.ceil(max(z["orbit"]) + 4 * z["glow_sigma"])
        region = dilate(zone, reach)
        return dict(kind="fireflies", label=z["label"], params=z, allowed=region,
                    zone=zone, mask=region, flies=flies, color=np.asarray(z["color"], np.float32))
    v0, v1 = z["visible_frames"]
    flies = []
    for _ in range(int(z["count"])):
        k = rng.integers(len(xs))
        flies.append(dict(
            x=float(xs[k]), y=float(ys[k]), r=float(rng.uniform(*z["orbit"])),
            ph=float(rng.uniform(0, 2 * np.pi)),
            on=set(rng.choice(sc.N, size=min(sc.N, int(rng.integers(v0, v1 + 1))),
                              replace=False).tolist())))
    reach = math.ceil(max(z["orbit"]) + 4 * z["glow_sigma"])
    region = dilate(zone, reach)
    return dict(kind="fireflies", label=z["label"], params=z, allowed=region,
                zone=zone, mask=region, flies=flies, color=np.asarray(z["color"], np.float32))


def fireflies_frame(sc, base, F, i):
    z = F["params"]
    out = base.copy()
    t = 2 * np.pi * i / sc.N
    cs, gs = z["core_sigma"], z["glow_sigma"]
    R = math.ceil(4 * gs)
    for f in F["flies"]:
        bright = 1.0
        if "bph" in f:  # плавный режим
            bright = (0.5 + 0.5 * math.cos(t + f["bph"])) ** z["blink_power"]
            bright = z["min_bright"] + (1 - z["min_bright"]) * bright
            if bright < 0.02:
                continue
            x = f["x"] + f["r"] * math.cos(f["sgn"] * t + f["ph"])
            y = f["y"] + z["orbit_y_ratio"] * f["r"] * math.sin(f["sgn"] * t + f["ph"])
        else:
            if i not in f["on"]:
                continue
            x = f["x"] + f["r"] * np.cos(t + f["ph"])
            y = f["y"] + z["orbit_y_ratio"] * f["r"] * np.sin(2 * t + f["ph"])
        x0, x1 = max(0, int(x) - R), min(sc.W, int(x) + R + 1)
        y0, y1 = max(0, int(y) - R), min(sc.H, int(y) + R + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        d2 = (sc.xx[y0:y1, x0:x1] - x) ** 2 + (sc.yy[y0:y1, x0:x1] - y) ** 2
        g = np.exp(-d2 / (2 * cs ** 2)) + z["glow_strength"] * np.exp(-d2 / (2 * gs ** 2))
        g = np.clip(bright * g, 0, 1)[..., None]
        win = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = win + (F["color"] - win) * g
    return out


# медленная пыль, плавно появляется и гаснет
def dust_setup(sc, z, idx):
    zone = polygon_mask(z["polygon"], sc.shape) if z.get("polygon") else np.ones(sc.shape, bool)
    zone &= ~polys_mask(z.get("exclude"), sc.shape)
    if not zone.any():
        return None
    rng = sc.rng("dust", idx)
    pr = zone.astype(np.float64)
    lum = gaussian_filter(sc.lum, 6)
    if z.get("light"):
        lo, hi = z["light"]
    else:
        lo, hi = np.percentile(lum[zone], [10, 95])
    pr *= np.clip((lum - lo) / max(hi - lo, 1), 0.03, 1) ** z["light_bias"]
    pr = pr.ravel() / pr.sum()
    if z["color"] == "auto":
        thr = np.percentile(sc.lum[zone], 97)
        col = sc.img[zone & (sc.lum >= thr)].mean(0)
        col = col + 0.5 * (255 - col)
    else:
        col = np.asarray(z["color"], np.float32)
    n = int(z["count"])
    kk = rng.choice(sc.H * sc.W, n, p=pr)
    d0 = math.atan2(z["direction"][1], z["direction"][0])
    ang = d0 + np.deg2rad(z["spread"]) * rng.uniform(-1, 1, n)
    soft = rng.random(n) < z["soft_share"]
    spd = rng.uniform(*z["speed"], n)
    parts = dict(x=(kk % sc.W).astype(np.float32), y=(kk // sc.W).astype(np.float32),
                 vx=np.cos(ang) * spd, vy=np.sin(ang) * spd,
                 ph=rng.uniform(0, sc.N, n), wph=rng.uniform(0, 2 * np.pi, n),
                 size=np.where(soft, rng.uniform(*z["soft_size"], n), rng.uniform(*z["size"], n)),
                 op=np.where(soft, rng.uniform(*z["soft_opacity"], n), rng.uniform(*z["opacity"], n)))
    return dict(kind="dust", label=z["label"], params=z, zone=zone, mask=zone, allowed=zone,
                parts=parts, color=col.astype(np.float32))


def dust_frame(sc, base, P, i):
    z, p, N = P["params"], P["parts"], sc.N
    a_map = np.zeros(sc.shape, np.float32)
    for k in range(len(p["x"])):
        u = ((i + p["ph"][k]) % N) / N
        env = math.sin(math.pi * u) ** (2 * z["fade_power"])
        if env < 0.02:
            continue
        s_ = (u - 0.5) * N
        wob = z["wobble"] * math.sin(2 * math.pi * u + p["wph"][k])
        x = p["x"][k] + p["vx"][k] * s_ - p["vy"][k] * wob * 0.5
        y = p["y"][k] + p["vy"][k] * s_ + p["vx"][k] * wob * 0.5
        sg = p["size"][k]
        R = math.ceil(3 * sg) + 1
        x0, x1 = max(0, int(x) - R), min(sc.W, int(x) + R + 1)
        y0, y1 = max(0, int(y) - R), min(sc.H, int(y) + R + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        g = np.exp(-((sc.xx[y0:y1, x0:x1] - x) ** 2 + (sc.yy[y0:y1, x0:x1] - y) ** 2) / (2 * sg * sg))
        a_map[y0:y1, x0:x1] = np.maximum(a_map[y0:y1, x0:x1], env * p["op"][k] * g)
    a = (a_map * P["zone"])[..., None]
    return base + (P["color"] - base) * a


# глаза светятся и пульсируют
def eyes_setup(sc, z, idx):
    zone, m = object_mask(sc, z)
    m = remove_small(morph(m, cv2.MORPH_CLOSE, 1), 3)
    if not m.any():
        return None
    hole = dilate(m, z["pad"])
    plate = cv2.inpaint(sc.u8, hole.astype(np.uint8) * 255, 4, cv2.INPAINT_TELEA).astype(np.float32)
    lab, n = label(hole)
    yc = np.zeros(sc.shape, np.float32)
    hh = np.ones(sc.shape, np.float32)
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        yc[lab == k] = ys.mean()
        hh[lab == k] = (ys.max() - ys.min() + 1) / 2
    rel = np.abs(sc.yy - yc) / hh
    col = sc.img[m].mean(0)
    halo = gaussian_filter(m.astype(np.float32), z["halo_sigma"])
    halo /= halo.max() + 1e-6
    region = dilate(hole, int(3 * z["halo_sigma"]))
    return dict(kind="eyes", label=z["label"], params=z, zone=zone, mask=m, allowed=region,
                hole=hole, plate=plate, rel=rel, col=col, halo=halo, region=region,
                soft=gaussian_filter(hole.astype(np.float32), 0.7))


def eyes_frame(sc, base, E, i):
    z = E["params"]
    o, g = z["open"][i % len(z["open"])], z["glow"][i % len(z["glow"])]
    vis = np.clip((o - E["rel"]) / z["lid_soft"] * 3 + 0.5, 0, 1) if o > 0 else np.zeros(sc.shape, np.float32)
    vis = (vis * E["soft"] + (1 - E["soft"]))[..., None]
    lit = base * (1 + g * E["soft"][..., None])
    out = E["plate"] * (1 - vis) + lit * vis
    out = out + (z["halo_strength"] * (o * (0.6 + g)) - z["halo_strength"] * 0.6) \
        * E["halo"][..., None] * E["col"] * 0.5
    return np.where(E["region"][..., None], np.clip(out, 0, 255), base)


# эффект: (подготовка, кадр). в таком порядке и рисуются
EFFECTS = {
    "water": (water_setup, water_frame),
    "wind": (wind_setup, wind_frame),
    "clouds": (clouds_setup, clouds_frame),
    "rays": (rays_setup, rays_frame),
    "light": (light_setup, light_frame),
    "glow": (glow_setup, glow_frame),
    "eyes": (eyes_setup, eyes_frame),
    "mist": (mist_setup, mist_frame),
    "plants": (plant_setup, None),  # растения рисуются все вместе, у них общая подложка
    "flap": (flap_setup, flap_frame),
    "particles": (particles_setup, particles_frame),
    "dust": (dust_setup, dust_frame),
    "fireflies": (fireflies_setup, fireflies_frame),
}


# готовит все эффекты локации
def build(sc, cfg):
    st = {}
    for key, (setup, _) in EFFECTS.items():
        st[key] = [e for k, z in enumerate(cfg.get(key, [])) if (e := setup(sc, z, z.get("_idx", k)))]
        if key == "water":  # маска всей воды, нужна для отблесков рядом с водой
            sc.water_mask = np.zeros(sc.shape, bool)
            for w in st["water"]:
                sc.water_mask |= w["mask"]
    st["plants_all"] = plants_combine(sc, st["plants"])
    return st


# для баланса: что на картинке главное, а что фон
ROLE = {"water": "object", "plants": "object", "glow": "object", "eyes": "object", "flap": "object",
        "fireflies": "object", "particles": "object", "rays": "object",  # лучи тоже объект, они нарисованы
        "light": "ambient", "dust": "ambient", "wind": "ambient", "clouds": "ambient", "mist": "ambient"}


# накладывает эффект с учётом силы из баланса
def _apply(sc, f, key, e, frame, i):
    nf = frame(sc, f, e, i)
    g, km = e.get("gain", 1.0), e.get("keep")
    if key == "flap" or (g == 1.0 and km is None):
        return nf
    w = g if km is None else g * km
    return f + (nf - f) * w


def render_frames(sc, st, only=None):
    frames = []
    for i in range(sc.N):
        f = sc.img.copy()
        for key, (_, frame) in EFFECTS.items():
            if only is not None and key not in only:
                continue
            if key == "plants":
                if st["plants_all"]:
                    f = plants_frame(sc, f, st["plants_all"], i)
                continue
            for e in st[key]:
                f = _apply(sc, f, key, e, frame, i)
        frames.append(np.clip(np.rint(f), 0, 255).astype(np.uint8))
    return frames


# сколько всего дёргается между кадрами, одним числом
def _motion_energy(frames, clip):
    L = [f.astype(np.float32).mean(-1) for f in frames]
    d = np.max([np.abs(L[i] - L[(i + 1) % len(L)]) for i in range(len(L))], 0)
    return float(np.minimum(d, clip).mean())


# тут считаем сколько всё дёргается и глушим фон,
# если объектов и так хватает
def balance(sc, st, bz):
    keys = [k for k in EFFECTS if st[k]]
    if not keys:
        return {}
    # фон не лезет туда, где главные объекты
    obj = np.zeros(sc.shape, bool)
    for k in keys:
        if ROLE[k] == "object" and k not in ("fireflies", "particles"):
            if k == "plants":
                for p in st["plants"]:
                    obj |= p["w"] > 0.05
            else:
                for e in st[k]:
                    obj |= e["allowed"]
    keep = None
    if obj.any():
        keep = (1 - np.clip(gaussian_filter(obj.astype(np.float32), bz["keepout_blur"]) * 1.5, 0, 1))[..., None]
        for k in keys:
            if ROLE[k] == "ambient":
                for e in st[k]:
                    e["keep"] = keep
    E = {k: _motion_energy(render_frames(sc, st, only={k}), bz["clip"]) for k in keys}
    EO = sum(E[k] for k in keys if ROLE[k] == "object")
    EA = sum(E[k] for k in keys if ROLE[k] == "ambient")
    gO, gA = 1.0, 1.0
    if EO > bz["obj_max"]:
        gO = bz["obj_max"] / EO
    elif 0 < EO < bz["obj_min"]:
        gO = min(bz["obj_gain_max"], bz["obj_min"] / EO)
    EOs = EO * gO
    amb = [k for k in keys if ROLE[k] == "ambient"]
    gains = {k: gO for k in keys if ROLE[k] == "object"}
    if EA > 0:
        g_lo, g_hi = bz["amb_gain"]
        if EOs >= bz["obj_min"] and EOs + EA >= bz["total_min"]:
            # объект сильный: фон только тише, не громче ambient_ratio от объекта
            target = min(EA, max(bz["ambient_ratio"] * EOs, bz["total_min"] - EOs))
            g_hi = min(g_hi, 1.0)
        else:
            # объекта нет или он слабый: подтягиваем до total_min, режем выше ambient_only_max
            target = float(np.clip(EA, bz["total_min"] - EOs, bz["ambient_only_max"] - EOs))
        target = max(target, 0.0)
        pr = bz["priority"]

        def amb_gains(c):  # ветер глушим первым, свет и пыль последними
            return {k: float(np.clip(c * pr.get(k, 1.0), g_lo, g_hi)) for k in amb}

        lo, hi = 0.0, 50.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if sum(E[k] * g for k, g in amb_gains(mid).items()) < target:
                lo = mid
            else:
                hi = mid
        gains.update(amb_gains((lo + hi) / 2))

    def set_gains():
        for k in keys:
            if k == "plants":
                for p in st["plants"]:
                    p["gain"] = gains[k]
            else:
                for e in st[k]:
                    e["gain"] = gains[k]

    fixed = {k: float(v) for k, v in (bz.get("fixed") or {}).items() if k in gains}
    gains.update(fixed)  # что закреплено в конфиге локации, баланс не трогает
    set_gains()
    # перемеряем: мелкие изменения округляются в ноль, итог бывает меньше расчёта
    total = _motion_energy(render_frames(sc, st), bz["clip"])
    free = [k for k in amb if k not in fixed]
    if free and total < bz["total_min"] * bz["floor_tolerance"]:
        k_up = bz["total_min"] / max(total, 1e-3)
        for k in free:
            gains[k] = float(min(bz["amb_gain"][1], gains[k] * k_up))
        set_gains()
    return dict(energy={k: round(v, 3) for k, v in E.items()},
                gains={k: round(v, 3) for k, v in gains.items()}, measured=round(total, 3))


# карта: где кадры отличаются от оригинала
def change_map(sc, frames, z):
    diff = np.zeros(sc.shape, np.int16)
    for f in frames:
        diff = np.maximum(diff, np.abs(f.astype(np.int16) - sc.u8.astype(np.int16)).max(-1))
    gray = sc.img.mean(-1, keepdims=True).repeat(3, -1) * z["dim"]
    heat = cv2.applyColorMap(np.clip(diff * 255.0 / z["scale_max"], 0, 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)[..., ::-1].astype(np.float32)
    heat = np.maximum(heat, np.asarray(z["min_color"], np.float32))
    arr = np.where((diff > 0)[..., None], heat, gray)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# одна локация целиком: кадры, webp и карта изменений
def render(cfg, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_im = Image.open(cfg["source"])
    icc = src_im.info.get("icc_profile")
    img = np.asarray(src_im.convert("RGB"))
    sc = Scene(img, cfg)
    st = build(sc, cfg)
    bal = balance(sc, st, cfg["balance"]) if cfg["balance"]["enabled"] else {}
    frames = render_frames(sc, st)
    ims = [Image.fromarray(f) for f in frames]
    extra = {"icc_profile": icc} if icc else {}
    for i, im in enumerate(ims):
        im.save(out_dir / f"frame_{i + 1}.png", **extra)
    wp = cfg["webp"]
    ims[0].save(out_dir / "anim.webp", save_all=True, append_images=ims[1:],
                duration=cfg["frame_ms"], loop=0, quality=wp["quality"],
                method=wp["method"], lossless=wp["lossless"],
                minimize_size=wp.get("minimize_size", False), **extra)
    change_map(sc, frames, cfg["change_map"]).save(out_dir / "change_map.png")
    return sc, st, bal, frames


