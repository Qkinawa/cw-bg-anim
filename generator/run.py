import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# по одному потоку на процесс, иначе процессы толкаются за ядра
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import cv2
import locanim

HERE = Path(__file__).resolve().parent


# один фон, каждый в своём процессе
def render_one(config, image, out, defaults):
    cv2.setNumThreads(1)
    cfg = locanim.load_config(config, defaults)
    cfg["name"] = image.stem
    cfg["source"] = str(image)
    locanim.render(cfg, out / image.stem)
    return image.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--locations", default=HERE / "locations", type=Path)
    ap.add_argument("--configs", default=HERE / "configs", type=Path)
    ap.add_argument("--out", default=HERE / "out", type=Path)
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    defaults = locanim.load_defaults(args.configs / "defaults.json")
    jobs = []
    for image in sorted(args.locations.glob("*.jpg")):
        config = args.configs / f"{image.stem}.json"
        if config.exists() and (not args.ids or image.stem in args.ids):
            jobs.append((config, image))
    # тяжёлые конфиги вперёд, чтоб в конце не ждать одну долгую локацию
    jobs.sort(key=lambda j: -j[0].stat().st_size)

    start = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(jobs)))) as pool:
        futures = [pool.submit(render_one, c, i, args.out, defaults) for c, i in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            print(f"[{done}/{len(jobs)}] {future.result()}")
    print(f"{len(jobs)} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
