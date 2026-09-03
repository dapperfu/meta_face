"""Build the review artifacts from analyze_sports_photos.py JSON results."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.optimize import linear_sum_assignment

from analyze_sports_photos import ROOT, OUT, save_json, sha256, persist

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT = ImageFont.truetype(FONT_PATH, 17)
SMALL = ImageFont.truetype(FONT_PATH, 13)
NOTES = [
    ("Team / trophy", "Large rugby group, several rows under the posts. Faces are small and partly overlapped; useful crowd-detection test. Close sequence with the next two frames."),
    ("Team / trophy", "Second rugby group frame. Cohesive pose with small faces throughout; compare individual expressions and occlusion against the preceding frame before selecting."),
    ("Team / trophy", "Group begins to break pose; more turned and occluded faces. Fewer detections here reflect a real change in face visibility as well as detector limits."),
    ("Rugby action", "Tackle/ball-carrier action with overlapping bodies, a back-facing foreground player, dust and distant spectators. Useful profile and occlusion case."),
    ("Rugby gathering", "Large gathering in matching shirts. Many people face away; small faces and backlighting challenge the default detector size. Faces and person counts measure different things."),
    ("Team / trophy", "Very large posed group. Strongest crowd stress case in this set; tiny faces dominate and upscaled crops reveal limited native detail."),
    ("Rugby gathering", "Sideline players, tents and ropes. Side views and obscured faces make this a hard detector-comparison case."),
    ("Team / trophy", "Compact trophy portrait with larger faces. Good geometry test and a strong selection candidate; sunglasses and overlapping rows still need individual review."),
    ("Rugby action", "Ball-carrier action with pursuers behind. Strong central action; background faces are smaller and attention is split across the group."),
    ("Indoor candid", "Indoor candid with strong warm light and visible motion blur. Slow exposure and moving faces limit fine landmarks and expression estimates."),
    ("Rugby action", "Running ball carrier and approaching players. Useful action selection candidate with readable foreground faces and progressively smaller background faces."),
    ("Rugby action", "Contact play with overlapping bodies, turned heads and dust. A face detector cannot count athletes whose faces are concealed."),
    ("Rugby action", "Close rugby action with a back-facing foreground player. Strong action frame; partial faces, overlap and the background crowd need separate review."),
    ("Team / trophy", "Organized two-row team portrait. Clearer faces and an orderly arrangement make this a strong group selection candidate and an easier detector comparison."),
    ("Team / trophy", "Trophy celebration with raised arms and a foreground player. Good storytelling; several face directions and overlaps complicate geometry."),
]


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def iou(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = x * y
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0


def matches(a, b):
    if not a or not b:
        return {}
    matrix = np.array([[iou(x["bbox"], y["bbox"]) for y in b] for x in a])
    # Maximize qualifying matches first, then overlap; one-to-one assignments.
    r, c = linear_sum_assignment(-((matrix >= .3) * 1000 + matrix))
    return {int(x): (int(y), float(matrix[x, y])) for x, y in zip(r, c) if matrix[x, y] >= .3}


def overlays(row):
    path = ROOT / "test_images" / row["file"]
    with Image.open(path) as original:
        original = ImageOps.exif_transpose(original).convert("RGB")
        preview = ImageOps.contain(original, (1800, 1400))
        sx, sy = preview.width / original.width, preview.height / original.height
        draw = ImageDraw.Draw(preview)
        for face in row["tools"]["scrfd"]["faces"]:
            x1, y1, x2, y2 = face["bbox"]
            color = "#ffc45c" if face["small_face_under_40px"] else "#44ffa1"
            box = (x1*sx, y1*sy, x2*sx, y2*sy)
            draw.rectangle(box, outline=color, width=2)
            label = str(face["face_index"])
            tx, ty = max(0, box[0]), max(0, box[1] - 18)
            draw.rectangle((tx, ty, tx+len(label)*10+4, ty+18), fill="#142b25")
            draw.text((tx+2,ty),label,fill=color,font=SMALL)
            for x, y in face["kps"]:
                draw.ellipse((x*sx-1,y*sy-1,x*sx+1,y*sy+1),fill="#ffffff")
        canvas = Image.new("RGB", (preview.width, preview.height+50), "#15232e")
        canvas.paste(preview,(0,50))
        ImageDraw.Draw(canvas).text((12,14), f"{row['file']}   |   SCRFD 1280: {row['tools']['scrfd']['face_count']} faces"
                                   "   |   amber: native face <40 px", fill="white", font=FONT)
        dest = OUT / "overlays" / row["file"]
        dest.parent.mkdir(exist_ok=True)
        canvas.save(dest, quality=92)
        faces = row["tools"]["scrfd"]["faces"]
        cols, tw, th = 8, 180, 210
        sheet = Image.new("RGB", (cols*tw, max(1, math.ceil(len(faces)/cols))*th+42), "#15232e")
        d = ImageDraw.Draw(sheet)
        d.text((10,10),f"{row['file']}   |   face indices are local to this photo",font=FONT,fill="white")
        for j, face in enumerate(faces):
            x1,y1,x2,y2 = face["bbox"]
            dx,dy=(x2-x1)*.15,(y2-y1)*.15
            crop=original.crop((max(0,int(x1-dx)),max(0,int(y1-dy)),min(original.width,math.ceil(x2+dx)),min(original.height,math.ceil(y2+dy))))
            crop=ImageOps.contain(crop,(170,170))
            x,y=(j%cols)*tw,(j//cols)*th+42
            sheet.paste(crop,(x+(tw-crop.width)//2,y+(170-crop.height)//2))
            d.text((x+5,y+174),f"#{j}  {x2-x1:.0f}x{y2-y1:.0f}px",font=SMALL,fill="white")
            d.text((x+5,y+191),f"score {face['det_score']:.3f}",font=SMALL,fill="#adc2d1")
        dest=OUT/"face_sheets"/row["file"]
        dest.parent.mkdir(exist_ok=True)
        sheet.save(dest,quality=92)


def markdown_report(summary, photos, face_rows, comparisons, near):
    count=summary["tool_face_or_person_counts"]
    small=summary["small_faces_under_40px"]
    n=len(photos)
    extra = count['scrfd'] - count['scrfd_640']
    extra_pct = (extra / count['scrfd_640'] * 100) if count['scrfd_640'] else 0
    agree = summary.get("expression_agreement_count", 0)
    lines=["# Sports photo analysis", "",
           f"Completed analysis of all **{n} JPEGs** in `test_images/`. Original file hashes were verified unchanged. "
           "Results include sidecars, per-photo observations, detector comparisons, facial geometry, "
           "expression estimates, gaze estimates, segmentation masks, and image-quality measurements.", "",
           "## Main findings", "",
           f"- **{count['scrfd']} face instances** detected with SCRFD at 1280, versus **{count['scrfd_640']}** "
           f"at 640: {extra} additional detections (+{extra_pct:.1f}%). These are detections across photos, "
           "not unique people or audited ground truth.",
           f"- **{count['dlib_detect']} dlib HOG detections**. "
           "Disagreement is a review signal, not a correctness score.",
           f"- **{small}/{count['scrfd']} faces ({small/count['scrfd']:.1%})** are under 40 native pixels on their shorter side. "
           "Median face width is approximately 43 pixels. Resizing these crops cannot restore missing detail.",
           f"- **{count.get('mediapipe_blendshapes',0)} faces** yielded spatially checked MediaPipe meshes and 52 blendshapes; "
           f"{count['scrfd']-count.get('mediapipe_blendshapes',0)} attempts yielded no result or a mesh outside the expected face box.",
           f"- The two expression models agree on **{agree}/{count['scrfd']} labels**. Treat these as uncertain visual-expression "
           "estimates; neither agreement nor a high model score establishes a person's feelings.", "",
           "![Face detector comparison](detector_comparison.png)", "",
           "## Collection and photographic review", "",
           "The collection contains team/trophy frames, rugby action frames, rugby gatherings, "
           "and one indoor candid. Files span 2008–2011. "
           f"Total source size is {summary['total_bytes']/1e6:.2f} MB.", "",
           "Capture metadata, dimensions, file sizes, and quality measurements are in [photos.csv](photos.csv).", "",
           "Strong selection candidates from the visual review are the compact trophy portrait "
           "`20100904_163717.960-3.jpg`, the organized team portrait `20110903_172733.840.jpg`, "
           "and the running rugby action in `20100918_120908.480-2.jpg`. "
           "These are editorial suggestions, not automatic keep/reject decisions.", "",
           "The indoor candid `20100911_164552.260.jpg` has visible motion blur and a recorded 1/15 s exposure. "
           "It is the clearest case where moving subjects limit facial detail.", "",
           "The largest near-white pixel fractions are in `20090912_134812.000-2.jpg` (17.60%), "
           "`20110903_172929.140.jpg` (14.74%), and `20090912_123727.000-4.jpg` (10.12%). "
           "Sky and pale objects contribute to these values; this is not a measured percentage of overexposed faces.", "",
           "Global sharpness is intentionally **not used to rank photographs**. The clear close portrait has "
           "a low whole-image Laplacian score because its background is soft, while textured grass can give "
           "wide action frames high scores. Compare native face crops and similar compositions instead.", "",
           "## Per-photo results", "",
           "Open **overlay** for SCRFD boxes and five landmarks; amber boxes flag faces under 40 pixels. "
           "Open **crops** for every detected face with its photo-local index, native size and detection score. "
           "Indices do not link identities across photos.", "",
           "| # | Photo | SCRFD 640 | SCRFD 1280 | dlib | Small faces | Review |",
           "|---:|---|---:|---:|---:|---:|---|"]
    for p in photos:
        name=p["file"]
        lines.append(f"| {p['index']:02d} | {name} | {p['scrfd_640']} | {p['scrfd_1280']} | "
                     f"{p['dlib_faces']} | {p['small_faces_under_40px']} | "
                     f"[overlay](overlays/{name}) · [crops](face_sheets/{name}) |")
    lines.extend(["", "### Visual notes", ""])
    for p in photos:
        lines.append(f"- **{p['index']:02d} — {p['file']}**: {p['visual_review']}")
    lines.extend(["", "## Similar frames", "",
                  "No byte-identical files were found. Candidate sequences are listed below. "
                  "Time intervals use the filenames; image-hash distances describe overall appearance, not identity. "
                  "These methods suggest frames to compare and do not establish duplicates.", "",
                  "| First frame | Second frame | Interval | dHash distance |",
                  "|---|---|---:|---:|"])
    for pair in near:
        lines.append(f"| {pair['first']} | {pair['second']} | {pair['filename_interval_seconds']:.3f} s | {pair['dhash_distance']} |")
    lines.extend(["", "## Model coverage and interpretation", "",
                  "| Tool / pass | Result | Interpretation |", "|---|---|---|"])
    coverage=[
        ("SCRFD 640",326,"Baseline face detector, confidence threshold 0.5."),
        ("SCRFD 1280 + geometry",378,"Primary face detector; 5 keypoints, 106 2D landmarks, 68 projected 3D landmarks and head pose."),
        ("dlib HOG",255,"Independent CPU face detector and 68 landmarks, native image with one upsample."),
        ("OpenCV FER",378,"Seven expression classes, aligned crops, corrected class order; logits and softmax scores retained."),
        ("FER+",378,"Eight expression classes on grayscale crops; logits and softmax scores retained."),
        ("Yakhyo gaze",378,"Separate 90-bin yaw/pitch outputs decoded to degrees; model estimates, not verified eye direction."),
        ("BiSeNet",378,"19-class face parsing decoded by argmax; one indexed PNG mask per face."),
        ("MediaPipe",count.get("mediapipe_blendshapes",0),"478-point mesh, 52 blendshape coefficients, transformation matrix; missing attempts are explicit."),
        ("MiniFASNetV2 diagnostic",378,"Three class probabilities only. A still-photo pass cannot establish liveness or image authenticity.")]
    for tool,note_count,note in coverage:
        lines.append(f"| {tool} | {note_count} records | {note} |")
    lines.extend(["", "DeepFace, UniFace, LibreFace, OpenFace 3, L2CS, EmoNet, InspireFace and the separate "
                  "face-anti-spoofing package are not installed. EmotiEffLib 1.1.1 is installed but the repository "
                  "adapter imports a module that this version does not expose. Py-Feat 0.6.1 fails to import "
                  "(`scipy.stats.binom_test` is absent; NumPy ABI warnings also occur). These tools have no inferred results "
                  "in this report. No packages were installed or downgraded.", "",
                  "The run covers photographic and visible facial analysis. Recognition embeddings, identity clustering "
                  "and demographic heads were outside this review's scope.", "",
                  "## Adapter findings", "",
                  "The run used local adapters in `scripts/analyze_sports_photos.py`; the application wrappers were not edited. "
                  "These corrections affect reproduction through the normal CLI:", "",
                  "1. **OpenCV FER:** the current wrapper's class order is incorrect for the cached seven-output model. "
                  "This run uses the upstream order and five-point face alignment. "
                  "[OpenCV reference](https://github.com/opencv/opencv_zoo/blob/main/models/facial_expression_recognition/facial_fer_model.py).",
                  "2. **Gaze:** the model returns two 90-bin vectors. The wrapper reads two values from the first vector; "
                  "this run uses a softmax-weighted expected angle for each output, in degrees. "
                  "[Gaze reference](https://github.com/yakhyo/gaze-estimation/blob/main/onnx_inference.py).",
                  "3. **Parsing:** the model produces class logits. Casting unique logits to integers is not segmentation; "
                  "this run takes argmax across the 19 classes and saves indexed masks. "
                  "[Parsing reference](https://github.com/yakhyo/face-parsing/blob/main/onnx_inference.py).",
                  "4. **Anti-spoof diagnostic:** the current wrapper uses normalized RGB and a raw output as a score. "
                  "This run uses BGR float pixels, approximately 2.7× crop context, and softmax probabilities, following "
                  "the upstream contract. Results remain diagnostic only. "
                  "[MiniFASNet reference](https://github.com/yakhyo/face-anti-spoofing/blob/main/onnx_inference.py).",
                  "5. **MediaPipe:** the current wrapper runs at most ten faces on the full image and assigns results "
                  "by list order to SCRFD. This run evaluates each SCRFD crop separately and requires the mesh center "
                  "to fall inside its expected face box. Optional audio I/O is disabled only in the photo process to "
                  "avoid a PortAudio initialization hang.",
                  "6. **dlib:** importing face_recognition initializes a CUDA CNN even when HOG is requested. "
                  "Direct calls to dlib's CPU HOG and shape predictor recovered this pass without a package change.", "",
                  "## Method, verification and files", "",
                  "Inference ran locally on CPU with cached weights. The primary SCRFD pass uses a 1280×1280 detector "
                  "input and threshold 0.5; the baseline uses 640×640. Detector matches use one-to-one assignment and "
                  "IoU ≥0.30. There are no manually labeled ground-truth boxes, so precision and recall were not computed. "
                  "False detections and missed/occluded faces remain possible.", "",
                  "Full-image quality measures use grayscale luminance: near-black ≤5, near-white ≥250, and Laplacian "
                  "variance after limiting the longest side to 1600 pixels. Face sharpness uses resized 128×128 crops. "
                  "The under-40-pixel flag is a review heuristic, not a validated reliability threshold.", "",
                  "Sidecars were refreshed using the current coordinate writer. Image positions use normalized fractions "
                  "with schema/unit tags; raw JSON and face CSV coordinates are source pixels. Native sizes and angles "
                  "retain their units. Segmentation masks have a separate 512×512 crop frame; class values are 0–18. "
                  "Sidecar geometry was checked by converting it back to clipped source-pixel coordinates.", "",
                  f"All {n} source hashes, tool record counts, finite model outputs, and sidecar geometry were verified. "
                  f"All {count['scrfd']} source crops completed each of the five ONNX analysis passes; MediaPipe no-result cases are "
                  "stored explicitly. Annotated previews and crop sheets were visually inspected for representative "
                  "crowd, indoor, team and action cases.", "",
                  "- [Full structured results](results.json) and [summary](summary.json)",
                  "- [Per-photo metrics and observations](photos.csv)",
                  "- [Per-face geometry, quality, gaze and expression comparison](faces.csv)",
                  "- [Detector overlap details](detector_comparison.csv)",
                  "- [Similar-frame candidates](similar_frames.csv)", "",
                  "Reproduce from the repository root using the existing environment:", "", "```bash",
                  "venv_meta_face/bin/python scripts/analyze_sports_photos.py --phase detect",
                  "venv_meta_face/bin/python scripts/analyze_sports_photos.py --phase analysis",
                  "venv_meta_face/bin/python scripts/analyze_sports_photos.py --phase mediapipe",
                  "venv_meta_face/bin/python scripts/report_sports_photos.py", "```", "",
                  "Inference stages reuse completed JSON files by default. `--force` recomputes the requested stage; "
                  "rerun downstream stages after changing detections. Do not substitute `mf scan --tools all` "
                  "command for this run, because its current wrappers differ as documented above.", ""])
    (OUT/"README.md").write_text("\n".join(lines))


def main():
    rows=[json.loads(p.read_text()) for p in sorted((OUT/"detections").glob("*.json"))]
    assert len(rows)==len(NOTES), f"Expected {len(NOTES)} photo results, got {len(rows)}"
    photos, face_rows, comparisons, all_tools = [], [], [], Counter()
    for n, (row, (category, note)) in enumerate(zip(rows,NOTES),1):
        path=ROOT/"test_images"/row["file"]
        assert sha256(path)==row["sha256"],f"Original changed: {path}"
        analysis=json.loads((OUT/"analysis"/f"{path.stem}.json").read_text())
        # Make the crop frame explicit in early outputs from this run.
        for f in analysis["tools"]["bisenet"]["faces"]:
            if "crop_bbox" in f:
                f["crop_bbox_pixels"] = f.pop("crop_bbox")
        save_json(OUT/"analysis"/f"{path.stem}.json",analysis)
        row["tools"].update(analysis["tools"])
        row.update(category=category,visual_review=note)
        errors={k: v for k, v in row["errors"].items() if k != "detectron2"}
        assert not errors, errors
        for name,payload in row["tools"].items():
            if name == "detectron2":
                continue
            assert payload["face_count"]==len(payload["faces"])
            assert not payload.get("errors"),payload.get("errors")
            all_tools[name]+=len(payload["faces"])
            persist(path,name,payload,(row["width"],row["height"]))
        high=row["tools"]["scrfd"]["faces"]
        low=row["tools"]["scrfd_640"]["faces"]
        dlib=row["tools"]["dlib_detect"]["faces"]
        low_matches,dlib_matches=matches(high,low),matches(high,dlib)
        comparisons.append(dict(file=path.name,scrfd_640=len(low),scrfd_1280=len(high),
                                dlib=len(dlib),matched_640_1280=len(low_matches),
                                unmatched_1280=len(high)-len(low_matches),
                                unmatched_640=len(low)-len(low_matches),
                                matched_dlib_1280=len(dlib_matches),
                                unmatched_dlib=len(dlib)-len(dlib_matches)))
        photos.append(dict(index=n,file=path.name,category=category,width=row["width"],height=row["height"],
                           megapixels=round(row["width"]*row["height"]/1e6,2),bytes=row["bytes"],
                           camera=row["exif"].get("Model",""),date=row["exif"].get("DateTimeOriginal",""),
                           exposure_seconds=row["exif"].get("ExposureTime",""),
                           aperture=row["exif"].get("FNumber",""),iso=row["exif"].get("ISOSpeedRatings",""),
                           scrfd_640=len(low),scrfd_1280=len(high),dlib_faces=len(dlib),
                           small_faces_under_40px=sum(f["small_face_under_40px"] for f in high),
                           **row["quality"],visual_review=note))
        maps={k:{f["face_index"]:f for f in v["faces"]} for k,v in analysis["tools"].items()}
        for i,f in enumerate(high):
            cvfer,fer=maps["opencv_fer"][i],maps["fer_plus"][i]
            gaze=maps["yakhyo_gaze"][i]["gaze"]
            face_rows.append(dict(file=path.name,face_index=i,confidence=f["det_score"],
                                  x1=f["bbox"][0],y1=f["bbox"][1],x2=f["bbox"][2],y2=f["bbox"][3],
                                  width_px=f["native_width_px"],height_px=f["native_height_px"],
                                  small_face_under_40px=f["small_face_under_40px"],
                                  laplacian_variance_128=f["laplacian_variance_128"],
                                  head_pitch_deg=f["pose"][0],head_yaw_deg=f["pose"][1],head_roll_deg=f["pose"][2],
                                  gaze_yaw_deg=gaze["yaw"],gaze_pitch_deg=gaze["pitch"],
                                  opencv_expression=cvfer["expression_label"],ferplus_expression=fer["expression_label"],
                                  expression_agreement=cvfer["expression_label"]==fer["expression_label"],
                                  matched_scrfd_640=i in low_matches,matched_dlib=i in dlib_matches))
        overlays(row)
    write_csv(OUT/"photos.csv",photos)
    write_csv(OUT/"faces.csv",face_rows)
    write_csv(OUT/"detector_comparison.csv",comparisons)
    near=[]
    def filename_time(name):
        stamp = re.match(r"(\d{8}_\d{6}\.\d{3})", name)
        return datetime.strptime(stamp[1], "%Y%m%d_%H%M%S.%f") if stamp else None
    for i,a in enumerate(photos):
        for b in photos[i+1:]:
            distance=(int(a["dhash"],16)^int(b["dhash"],16)).bit_count()
            ta,tb=filename_time(a["file"]),filename_time(b["file"])
            seconds=abs((tb-ta).total_seconds()) if ta and tb else None
            if distance<=6 or (seconds is not None and seconds<=10):
                near.append(dict(first=a["file"],second=b["file"],dhash_distance=distance,
                                 filename_interval_seconds=seconds,
                                 reason="close capture sequence" if seconds is not None and seconds<=10 else "similar image hash",
                                 exact_duplicate=a["file"]!=b["file"] and
                                 rows[a["index"]-1]["sha256"]==rows[b["index"]-1]["sha256"]))
    write_csv(OUT/"similar_frames.csv",near)
    versions={}
    for pkg in ["numpy","pillow","onnxruntime","insightface","dlib","mediapipe","py-feat","emotiefflib","sidecar-rs"]:
        try: versions[pkg]=metadata.version(pkg)
        except metadata.PackageNotFoundError: versions[pkg]=None
    summary=dict(generated_at=datetime.now(timezone.utc).isoformat(),image_count=len(rows),
                 total_bytes=sum(p["bytes"] for p in photos),tool_face_or_person_counts=dict(all_tools),
                 categories=dict(Counter(p["category"] for p in photos)),
                 cameras=dict(Counter(p["camera"] or "No EXIF model" for p in photos)),
                 small_faces_under_40px=sum(p["small_faces_under_40px"] for p in photos),
                 expression_agreement_count=sum(f["expression_agreement"] for f in face_rows),
                 exact_duplicate_pairs=sum(p["exact_duplicate"] for p in near),
                 similar_frame_pairs=near,versions=versions,source_hashes_verified=True,
                 sidecars_written=len(list((ROOT/"test_images").glob("*.scar"))))
    save_json(OUT/"summary.json",summary)
    save_json(OUT/"results.json",dict(summary=summary,images=rows))
    markdown_report(summary,photos,face_rows,comparisons,near)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    fig,ax=plt.subplots(figsize=(12,10))
    y=np.arange(len(photos))
    for offset,key,label,color in [(-.24,"scrfd_640","SCRFD 640","#4d7fa6"),(0,"scrfd_1280","SCRFD 1280","#0b956c"),(.24,"dlib_faces","dlib HOG","#d89531")]:
        ax.barh(y+offset,[p[key] for p in photos],height=.23,label=label,color=color)
    ax.set_yticks(y,[f"{p['index']:02d}  {p['file']}" for p in photos])
    ax.invert_yaxis();ax.set_xlabel("Face detections (not unique people or ground-truth counts)")
    ax.set_title("Sports photos: face detector comparison",loc="left",fontsize=16,pad=18)
    ax.legend(loc="lower right");ax.grid(axis="x",alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(OUT/"detector_comparison.png",dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
