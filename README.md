# Express-Derm ML

Reproducible dataset curation, patient-safe splitting, model training,
calibration, evaluation, ONNX export, and C++/TensorRT inference for
Express-Derm.

> **Research only:** this repository produces screening-model artifacts, not a
> diagnosis. Results from one dermoscopy collection do not establish
> generalization to another collection.

## Two supported paths

| Path | Purpose | Where it runs |
| --- | --- | --- |
| Python / PyTorch | curation, grouped split, training, calibration, metrics, ONNX export | GPU workstation, Apple Silicon, or CPU |
| C++ / TensorRT | persistent low-latency inference from an exported ONNX graph | compatible Linux/NVIDIA GPU system |

C++ is appropriate for deployment inference. The training pipeline intentionally
remains in Python: PyTorch and the scientific Python stack provide the audited
data, metrics, calibration, and reproducibility tooling used by this project.
Reimplementing training in C++ would create a second scientific pipeline to
validate without improving the deployed model.

## Repository layout

~~~text
configs/              versioned datasets and experiments
express_derm_ml/      Python pipeline
tests/                synthetic tests; no clinical images
cpp/ai_worker/        persistent TensorRT 10 C++ worker
scripts/              TensorRT engine and worker build scripts
requirements*.txt     data, development, and training environments
~~~

Datasets, generated manifests, runs, checkpoints, models, and TensorRT engines
are excluded from Git.

## Quick start

Prerequisites are Python 3.11 or newer, pip, and GNU Make. Using an activated
Python virtual environment is recommended; choose its name and location
locally. The project does not assume a particular environment path.

For tests and data-pipeline development:

~~~bash
git clone https://github.com/mpatrini7/Express-derm-ml.git
cd Express-derm-ml
python3 -m pip install -r requirements-dev.txt
make check
~~~

For training, install PyTorch, ONNX, and the full pipeline:

~~~bash
python3 -m pip install -r requirements.txt
~~~

The default device policy selects CUDA first, Apple Metal (MPS) second, and CPU
last. Pass --device cuda, --device mps, or --device cpu to require one backend;
an unavailable requested accelerator fails explicitly.

## High-resolution experiments

The selected `express-derm-1` research artifact has a fixed `224x224` input and
uses the historical stretch-to-square preprocessing contract. That baseline is
kept immutable. New high-resolution experiments use a separate contract that:

- keeps the original aspect ratio;
- centers the image on ImageNet-mean padding instead of deforming it;
- trains and evaluates at either `512x512` or `1024x1024`;
- records the preprocessing version in the checkpoint, run, calibration,
  metrics, ONNX manifest, and parity report;
- supports gradient accumulation so the effective batch does not depend on
  fitting many 1024-pixel images in memory at once.

Measure the local cost before a full run:

~~~bash
python3 -m express_derm_ml.benchmark_resolution \
  --architecture efficientnet_b0 \
  --sizes 224,512,1024 \
  --batch-size 1 \
  --device auto
~~~

Start with the 512-pixel candidate:

~~~bash
python3 -m express_derm_ml.train \
  --config configs/v25_efficientnet_b0_512_letterbox_highres.yaml \
  --manifest artifacts/training/manifest_split.csv \
  --images-dir data/combined-images.nosync \
  --output-dir runs/efficientnet-b0-v25-512-letterbox \
  --device auto
~~~

If that run passes the validation check, reuse its resolution-independent
weights to initialize the more expensive 1024-pixel stage:

~~~bash
python3 -m express_derm_ml.train \
  --config configs/v26_efficientnet_b0_1024_letterbox_highres.yaml \
  --manifest artifacts/training/manifest_split.csv \
  --images-dir data/combined-images.nosync \
  --initial-checkpoint runs/efficientnet-b0-v25-512-letterbox/best.pt \
  --output-dir runs/efficientnet-b0-v26-1024-letterbox \
  --device auto
~~~

The recorded v25 gate was executed on MPS and **failed**, so v26 was not
started. The best 512-pixel checkpoint selected at epoch 3 reached validation
PR-AUC `0.1512`, internal test ROC-AUC `0.8692` and PR-AUC `0.1630`, then
dropped to ROC-AUC `0.5636` and PR-AUC `0.1451` on locked MILK10k. At the
validation-frozen high threshold, MILK10k specificity was `0.3234` and
precision was `0.0860`. The comparable v23 224-pixel candidate remained better
on both the internal test and MILK10k. The v25 checkpoint is rejected and the
v26 configuration is retained only as an unexecuted research recipe, not as a
promotion candidate.

The 1024 input contains about 20.9 times as many pixels as 224. Higher
resolution does not by itself solve microscope-domain mismatch, so promotion
still requires the same patient-safe internal, external, and local microscope
gates used by every other candidate.

## Broad-attention v27 experiment

Increasing resolution alone did not improve generalization, so v27 tested a
different hypothesis: retain `224x224`, train over original, 90%, and 80%
center crops, and add same-source ISIC 2019 examples for BCC, SCC, and actinic
keratosis. This is a separate broad-malignancy attention target, not a new
meaning for the melanoma score produced by `express-derm-1`.

The added train-only set contains 4,818 BCC/SCC/AK images and 4,818 balanced
NV/BKL/DF/VASC controls, grouped by lesion. ISIC 2019 does not provide patient
identifiers for these records, so they are never used for validation or test.
The combined manifest has no exact split overlap and no perceptual candidate
crossing a split. All sources, hashes, licenses, selection roles, and the
unavailable-patient-ID status remain recorded in the artifacts.

The frozen v27 checkpoint has SHA-256
`395ccf58289e7ae95df74462a3bc4dcb8fe882c29da5fc8b536bb14dc18af0cf`.
It was initialized from `express-derm-1`, selected at epoch 7, and stopped early
after epoch 11. Its validation PR-AUC was `0.1261`, below the current model's
`0.1911`, so v27 is **rejected as a replacement**.

The center-scale consensus comparison makes the trade-off explicit:

| Frozen evaluation | Model / target | ROC-AUC | PR-AUC | Sensitivity | Specificity | Precision | FP / TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ISIC 2020 internal test (`n=6,538`) | `express-derm-1`, melanoma | 0.8986 | 0.2326 | 0.4870 | 0.9620 | 0.1867 | 244 / 56 |
| ISIC 2020 internal test (`n=6,538`) | v27 broad attention | 0.8936 | 0.1821 | 0.6522 | 0.9211 | 0.1289 | 507 / 75 |
| Decontaminated MILK10k (`n=5,231`) | `express-derm-1`, broad exploratory | 0.5115 | 0.7236 | 0.3964 | 0.6175 | 0.7248 | 565 / 1,488 |
| Decontaminated MILK10k (`n=5,231`) | v27 broad attention | 0.8268 | 0.9088 | 0.9038 | 0.5538 | 0.8374 | 659 / 3,393 |

MILK10k is external research evidence only. Nine images forming perceptual
cross-source candidates against the new training corpus were excluded before
evaluation; the retained set has zero exact or configured perceptual overlap.
The v27 center-scale policy marks 24.70% of the 745 retained NV examples high,
compared with 43.49% for the current model. A threshold preselected for 99%
validation specificity reduces that v27 NV rate to 4.70%, but on the internal
test it also reduces sensitivity to 19.13% and still yields only 21.36%
precision. That threshold is therefore not a safe standalone answer.

The research decision is to keep two distinct candidate signals:

- `express-derm-1` remains the melanoma-attention baseline;
- v27 remains an experimental broad-malignancy attention head;
- agreement may support a future **high-confirmed** tier, but disagreement or
  any non-high result must remain review/inconclusive rather than silently
  becoming low;
- a product may expose the combination only as an explicit research prototype;
  deployment authorization still requires an accepted microscope-domain set
  to pass frozen gates.

The frozen combined policy is versioned in
`configs/dual_model_policy_v1.yaml`. It requires all original, 90%-crop, and
80%-crop melanoma scores to be high and all three broad scores to exceed the
validation-selected `0.1613025932` confirmation threshold. Both models and all
views must be low before `no_elevated_signal` is emitted; every other case is
`review`.

On the internal test this produces 54 high-confirmed results: 18 true positives
and 36 false positives, for sensitivity `0.1565`, specificity `0.9944`, and
precision `0.3333`. On decontaminated MILK10k broad malignancy it produces 893
true positives and 82 false positives, for sensitivity `0.2379`, specificity
`0.9445`, and precision `0.9159`. The large sensitivity loss is intentional:
high-confirmed is an additional strict tier, never the only warning path.

Reproduce the decision artifact from the two immutable multi-view runs with:

~~~bash
python3 -m express_derm_ml.evaluate_dual_policy \
  --config configs/dual_model_policy_v1.yaml \
  --melanoma-report runs/melanoma/multiview-test.json \
  --melanoma-predictions runs/melanoma/multiview-test.npz \
  --broad-report runs/broad/multiview-test.json \
  --broad-predictions runs/broad/multiview-test.npz \
  --output runs/dual-policy/internal-test.json
~~~

The added data can be reproduced with
`download_isic2019_broad_malignancy`, `prepare_manifest`, and
`augment_training_manifest`, using
`configs/isic_2019_broad_attention_train.yaml`. The training recipe is
`configs/v27_efficientnet_b0_224_broad_attention_multiscale.yaml`; provide the
published `express-derm-1` checkpoint through `train --initial-checkpoint`.
External data is never authorized as training input.

## Dataset downloads

The baseline is fixed to **SIIM-ISIC 2020 Challenge Training**, collection 70,
under CC BY-NC 4.0. Keep the source files immutable and preserve their URLs,
SHA-256 receipts, license, attribution, and release identity.

Required baseline files:

| File | Official download |
| --- | --- |
| Training JPEG archive | [ISIC_2020_Training_JPEG.zip](https://isic-archive.s3.amazonaws.com/challenges/2020/ISIC_2020_Training_JPEG.zip) |
| Metadata v2 with lesion IDs | [ISIC_2020_Training_GroundTruth_v2.csv](https://isic-archive.s3.amazonaws.com/challenges/2020/ISIC_2020_Training_GroundTruth_v2.csv) |
| Official duplicate pairs | [ISIC_2020_Training_Duplicates.csv](https://isic-archive.s3.amazonaws.com/challenges/2020/ISIC_2020_Training_Duplicates.csv) |

Official collection references:

- [ISIC Challenge 2020 data](https://challenge.isic-archive.com/data/#2020)
- [ISIC Archive collection 70](https://api.isic-archive.com/collections/70/)
- [Dataset DOI 10.34970/2020-ds01](https://doi.org/10.34970/2020-ds01)

Suggested local layout:

~~~text
data/isic-2020.nosync/
  downloads/
    ISIC_2020_Training_JPEG.zip
    ISIC_2020_Training_GroundTruth_v2.csv
    ISIC_2020_Training_Duplicates.csv
    SHA256SUMS
  images/
    ISIC_0000000.jpg
    ...
  artifacts/
~~~

The JPEG archive is approximately 23 GB. Check free space before downloading
and extracting it. Do not combine JPEG and DICOM encodings as separate training
records; they represent the same cases.

Optional sources are kept separate and have additional eligibility rules:

| Dataset | Use | Official source |
| --- | --- | --- |
| ISIC 2019 patient-ID subset | train-only augmentation; only records with verified patient IDs | [2019 data](https://challenge.isic-archive.com/data/#2019), [collection 65](https://api.isic-archive.com/collections/65/) |
| ISIC-DICM-17K | patient-safe train-only research augmentation after overlap and pHash review | [collection 469](https://api.isic-archive.com/collections/469/), [DOI](https://doi.org/10.34970/233480) |
| Histopathology subset | train-only research source after patient/image exclusions and license checks | [ISIC collection 294](https://api.isic-archive.com/collections/294/) |
| MSK-1 benign controls | train-only hard-negative experiment; one CC0 image per lesion after overlap checks | [ISIC collection 289](https://api.isic-archive.com/collections/289/) |
| MILK10k | external evaluation only; never training because public patient IDs are absent | [MILK10k data](https://challenge.isic-archive.com/data/#milk10k), [DOI record](https://api.isic-archive.com/doi/milk10k/) |

The configuration files contain the canonical source, license, attribution,
expected counts, label mapping, and integrity policy for every supported input.
This repository does not redistribute dataset images.

## Improvement protocol: generalization first

Adding images is useful only when provenance, grouping, overlap, and the
held-out role remain explicit. The next model iteration uses this order:

1. keep MILK10k and every other external collection locked out of training;
2. assign patient-safe out-of-fold partitions within training-authorized
   sources;
3. train one fold model without the records it predicts;
4. mine hard false positives and false negatives only from those OOF
   predictions;
5. retrain a candidate with source-balanced sampling and the mined records;
6. apply one frozen threshold across the internal test and every completely
   held-out source;
7. reject the candidate if worst-source performance or cross-source gaps miss
   the versioned research gate.

This prevents the test set from gradually becoming training data and avoids
selecting a model merely because its internal aggregate metric improved.

`configs/v23_efficientnet_b2_224_multisource_priority_negatives_opencv.yaml`
records the first completed candidate for this protocol. It combines source-balanced sampling
across the historical and histopathology collections with guaranteed repeated
exposure to the 184 separately curated MSK-1 benign controls. The sampler keeps
the per-source class budget fixed, so prioritizing those negatives does not
silently enlarge an epoch or alter its positive balance.

The run stopped after 20 epochs with its best checkpoint at epoch 13. It was
rejected, not promoted: on the internal test its high-threshold sensitivity was
`0.5913`, specificity `0.9151`, and precision `0.1109`; on locked MILK10k they
were `0.5822`, `0.5979`, and `0.1197`. The formal gate also failed minimum
worst-source ROC-AUC, sensitivity, and specificity. The experiment shows that
repeating a small benign-control set improves some false-positive counts but
does not solve source generalization. The OOF mining stage is therefore the
next required input, not another increase in repeats.

The five-fold OOF experiment has now been executed on 22,736 development
records in 3,050 disjoint patient/lesion groups. Best fold ROC-AUC ranged from
`0.8344` to `0.9124` and PR-AUC from `0.4329` to `0.6159`. Every held-out
record was scored exactly once by a model that did not train on its group.
With predeclared raw-score thresholds `0.25/0.75`, mining found 1,215 severe
errors and retained 463 source-balanced examples: 234 hard false positives
and 229 hard false negatives. No test or external record was used.

The resulting v24 hard-example candidate was also executed and rejected. On
the locked internal test it reached ROC-AUC `0.8724`, PR-AUC `0.1979`, high
sensitivity `0.5739`, specificity `0.9080`, and precision `0.1005`. On locked
MILK10k it reached ROC-AUC `0.6611`, PR-AUC `0.1724`, high sensitivity
`0.6600`, specificity `0.5574`, and precision `0.1229`. These results do not
beat `express-derm-1` and do not pass the generalization objective.

A predeclared five-weight ensemble check selected 75% `express-derm-1` and
25% v24 using validation PR-AUC only. It improved internal and external
PR-AUC to `0.2906` and `0.2709`, respectively, but worsened the operational
external high decision: MILK10k specificity fell from `0.4881` to `0.4349`
and precision from `0.1333` to `0.1222`. The ensemble is therefore not a
promotion candidate. `express-derm-1` remains the selected research model.

### Create patient-safe OOF folds

~~~bash
python3 -m express_derm_ml.create_oof_plan \
  --manifest artifacts/training/manifest_split.csv \
  --output artifacts/training/oof_plan.csv \
  --folds 5 \
  --seed 2026
~~~

The plan is deterministic and stratified by source and group-level target. A
patient group is assigned to exactly one fold. The report stores hashes for the
held-out and training group sets of every fold; prediction provenance must
match those hashes.

Materialize, train, and calibrate each fold separately. For example:

~~~bash
python3 -m express_derm_ml.materialize_oof_fold \
  --manifest artifacts/training/manifest_split.csv \
  --plan artifacts/training/oof_plan.csv \
  --fold-id fold_0 \
  --output artifacts/training/oof_fold_0.csv

python3 -m express_derm_ml.train \
  --config configs/v23_efficientnet_b2_224_multisource_priority_negatives_opencv.yaml \
  --manifest artifacts/training/oof_fold_0.csv \
  --images-dir data/combined-images.nosync \
  --output-dir runs/oof/fold_0 \
  --device auto

python3 -m express_derm_ml.calibrate \
  --run-dir runs/oof/fold_0 \
  --images-dir data/combined-images.nosync \
  --device auto
~~~

Repeat for every planned fold, then collect exactly one validation prediction
set from each model:

~~~bash
python3 -m express_derm_ml.collect_oof_predictions \
  --plan artifacts/training/oof_plan.csv \
  --run-dir runs/oof/fold_0 \
  --run-dir runs/oof/fold_1 \
  --run-dir runs/oof/fold_2 \
  --run-dir runs/oof/fold_3 \
  --run-dir runs/oof/fold_4 \
  --output runs/oof/oof_predictions.npz \
  --provenance runs/oof/oof_provenance.json
~~~

After mining, train the hard-example candidate only with the verified OOF
artifact. The trainer rejects missing digests, external/test provenance,
records outside the final training split, and any mismatch in image name,
SHA-256, group, source, or target:

~~~bash
python3 -m express_derm_ml.train \
  --config configs/v24_efficientnet_b2_224_oof_hard_examples_opencv.yaml \
  --manifest artifacts/training/manifest_split.csv \
  --images-dir data/combined-images.nosync \
  --priority-records artifacts/training/oof_hard_errors.csv \
  --output-dir runs/efficientnet-b2-v24-oof-hard-examples \
  --device auto
~~~

The v24 policy always includes selected hard false negatives once per epoch
and repeats the union of verified hard false positives and the benign-control
collection twice. Epoch size and source balance remain fixed; priority records
do not add test or external images to training.

The collector verifies that every run trained on the exact complement of its
held-out patient groups. It intentionally collects uncalibrated sigmoid scores:
fold calibration may use held-out labels, while the mining score must remain a
model-only output.

### Mine errors without test leakage

After the fold models have produced a single `oof_predictions.npz` and its
provenance receipt, mine only the most severe errors per source:

~~~bash
python3 -m express_derm_ml.mine_oof_errors \
  --plan artifacts/training/oof_plan.csv \
  --predictions runs/oof/oof_predictions.npz \
  --provenance runs/oof/oof_provenance.json \
  --output artifacts/training/oof_hard_errors.csv \
  --low-threshold PREDECLARED_OOF_LOW_THRESHOLD \
  --high-threshold PREDECLARED_OOF_HIGH_THRESHOLD \
  --max-per-error-source 100
~~~

The command refuses ordinary validation, test, or external predictions. It
requires exact plan, prediction, fold-group, and checkpoint hashes. Its output
is a review/mining artifact; it never edits the source manifest.

### Run the multi-source gate

Copy `configs/generalization_gate.example.yaml`, fill in local artifact paths
and exact SHA-256 values, then run:

~~~bash
python3 -m express_derm_ml.generalization_gate \
  --config configs/generalization_gate.local.yaml \
  --output runs/candidate/generalization.json \
  --fail-on-gate
~~~

Every dataset is evaluated at the candidate's already frozen high threshold.
The report includes group-bootstrap intervals, PR-AUC lift over prevalence,
worst-source metrics, cross-source gaps, and explicit blockers. Passing this
gate advances a candidate only to the next research stage; it does not validate
deployment.

For the current `express-derm-1` artifact, the OpenCV/runtime-consistent audit
reproduces the following high-threshold results:

| Frozen set | ROC-AUC | PR-AUC | Sensitivity | Specificity | Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| ISIC 2020 internal test (`n=6,538`) | 0.9054 | 0.2500 | 0.5826 | 0.9466 | 0.1634 |
| MILK10k dermoscopic, fully held out (`n=5,240`) | 0.7066 | 0.2048 | 0.7067 | 0.5833 | 0.1374 |

The candidate fails the example research gate because worst-source specificity
is `0.5833` and remains far below its internal specificity. Earlier values of
`0.9061/0.2574` and `0.7462/0.2464` came from a preprocessing path inconsistent
with the published OpenCV runtime and must not be used for deployment
decisions. The generalization gap is the concrete problem the OOF hard-negative
and multi-source loop must improve; changing only the label shown by the
application would not fix it.

## Baseline pipeline

### 1. Build the attributed manifest

~~~bash
mkdir -p artifacts/isic-2020

python3 -m express_derm_ml.prepare_manifest \
  --metadata data/isic-2020.nosync/downloads/ISIC_2020_Training_GroundTruth_v2.csv \
  --images-dir data/isic-2020.nosync/images \
  --duplicate-pairs data/isic-2020.nosync/downloads/ISIC_2020_Training_Duplicates.csv \
  --config configs/isic_2020.yaml \
  --output artifacts/isic-2020/manifest.csv
~~~

The command verifies required metadata, expected release count, image
readability, dimensions, SHA-256, source license, attribution, exact duplicates,
and configured perceptual-hash candidates.

### 2. Create patient-grouped splits

~~~bash
python3 -m express_derm_ml.split_manifest \
  --manifest artifacts/isic-2020/manifest.csv \
  --output artifacts/isic-2020/manifest_split.csv
~~~

Every image from one patient or lesion group stays in one split. Source row
order does not change the canonical manifest identity or deterministic split.

### 3. Train

~~~bash
python3 -m express_derm_ml.train \
  --config configs/baseline.yaml \
  --manifest artifacts/isic-2020/manifest_split.csv \
  --images-dir data/isic-2020.nosync/images \
  --output-dir runs/efficientnet-b0-v1 \
  --device auto
~~~

Run directories are immutable: the pipeline refuses to reuse an existing
output directory. The preflight rechecks every image hash, split report, and
leakage gate before training.

### 4. Calibrate on validation only

~~~bash
python3 -m express_derm_ml.calibrate \
  --run-dir runs/efficientnet-b0-v1 \
  --images-dir data/isic-2020.nosync/images \
  --device auto
~~~

This fits affine logistic scaling and development attention thresholds using
only the validation split. Frozen development thresholds remain explicitly
unvalidated.

### 5. Evaluate the untouched test split

~~~bash
python3 -m express_derm_ml.evaluate \
  --run-dir runs/efficientnet-b0-v1 \
  --images-dir data/isic-2020.nosync/images \
  --device auto
~~~

Reports include ROC-AUC, PR-AUC, sensitivity, specificity, balanced accuracy,
precision, negative predictive value, Brier score, calibration error, and
grouped bootstrap intervals. Do not summarize this imbalanced task with a
single “success percentage”.

### 6. Export and verify ONNX

~~~bash
python3 -m express_derm_ml.export_onnx \
  --run-dir runs/efficientnet-b0-v1 \
  --output models/express-derm-1/model.onnx \
  --model-version express-derm-1

python3 -m express_derm_ml.verify_onnx \
  --run-dir runs/efficientnet-b0-v1 \
  --model-dir models/express-derm-1 \
  --images-dir data/isic-2020.nosync/images \
  --output runs/efficientnet-b0-v1/onnx_parity.json
~~~

The export directory contains the ONNX graph, runtime manifest, calibration,
metrics, dataset-manifest provenance, and hashes. Export success does not
promote validation or enable inference.

## C++ / TensorRT inference

Build TensorRT on the compatible Linux/NVIDIA GPU system that will run the
engine. Serialized TensorRT engines are not portable across arbitrary platform,
TensorRT, CUDA, and hardware combinations.

~~~bash
./scripts/build_tensorrt_engine.sh \
  models/express-derm-1/model.onnx \
  /tmp/express-derm-1.engine
~~~

The exported baseline graph has a fixed `1x3x224x224` input. Pass the optional
third argument only for an explicitly dynamic square `image` input.

Register the target-built engine without modifying the source ONNX package:

~~~bash
python3 -m express_derm_ml.register_tensorrt_engine \
  --model-dir models/express-derm-1 \
  --engine /tmp/express-derm-1.engine \
  --output-dir models/express-derm-1-gpu \
  --platform-version INSTALLED_PLATFORM_VERSION \
  --tensorrt-version INSTALLED_TENSORRT_VERSION \
  --device-model TARGET_GPU_MODEL \
  --precision fp16
~~~

Install the target C++ dependencies and compile the persistent worker:

~~~bash
sudo apt-get install build-essential cmake pkg-config libopencv-dev libssl-dev nlohmann-json3-dev
./scripts/build_cpp_ai_worker.sh
~~~

The worker uses TensorRT 10, CUDA, OpenCV, OpenSSL, and a local Unix socket. It
loads and hashes one registered model package at startup, keeps the engine,
execution context, stream, and buffers alive, and independently rejects
quality-rejected or source-unconfirmed requests. Acquisition-protocol state is
preserved as non-blocking context.

Its C++ output includes the raw logit, affine-calibrated score, binary-extremes
attention state, explicit abstention state, model and engine hashes, immutable
decision-policy version, and timings. C++ is a deployment path, not clinical
evidence.

## Integrity and leakage gates

The pipeline blocks:

- blank patient/group identifiers;
- patient or lesion groups spanning multiple splits;
- exact-image SHA-256 leakage;
- configured 256-bit pHash candidates crossing groups or splits;
- missing or changed image files;
- unknown labels or silent label remapping;
- unexpected source counts or exclusions;
- overwriting prior runs, calibration, metrics, models, or engines.

Perceptual duplicate screening uses ImageHash 4.3.2 pHash with hash size 16,
high-frequency factor 4, and Hamming distance 16. It is a curation heuristic,
not an image-quality or medical threshold.

## Validation gates

Keep deployment disabled until all of the following are documented:

- reproducible training from a versioned attributed manifest;
- zero patient/lesion leakage;
- frozen calibration and operating thresholds;
- at least one separately locked external dermoscopy source;
- PyTorch/ONNX/TensorRT numerical parity;
- C++ worker soak, latency, memory, power, and thermal measurements on target;
- proof that rejected or unapproved-source images cannot enter inference.

Every result must persist model version, model hash, thresholds, validation
status, domain status, backend, latency, and the complete raw result. Never
present an output as a diagnosis.
