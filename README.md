# Surf Maneuver Recognition

Recognition and temporal segmentation of surf maneuvers in video.

Final project for the Computer Vision course (PPGTI3007) at PPgTI/UFRN, 2026.2.

## Goal

Given a surf video, detect which maneuvers the surfer performs and when, producing a maneuver timeline (e.g. 0:12 take-off, 0:15 cutback, 0:19 roller).

## Approach

1. **Data preparation:** download, class balance analysis, clip statistics and frame sampling (16 evenly spaced frames per clip).
2. **Main model:** V-JEPA 2 (`facebook/vjepa2-vitl-fpc16-256-ssv2`) with a frozen video encoder and a new attentive probe trained for the 4 surf maneuvers.
3. **Comparison model:** VideoMAE fine-tuned on the same splits.
4. **Surfer localization:** YOLO26 detection and pose (Ultralytics), with LibreYOLO pose and SAM 3 text-prompted video tracking ("surfer") as alternatives. The clip is cropped around the surfer before classification, since the video processor keeps only the central 256x256 square.
5. **Long videos:** tracking + sliding window + merging consecutive windows into a maneuver timeline, with a "no maneuver" state for low-confidence windows.
6. **Web app:** Gradio interface (`gr.Video` input) that returns the annotated video and the timeline.

## Experiments

- V-JEPA 2 vs VideoMAE, trained with 100% and with 25% of the training data (data-efficiency comparison).
- Classification with and without the surfer crop.
- Temporal order test: the same clip played forward and reversed.
- Out-of-distribution test on own or Creative Commons surf videos, with maneuver times annotated in Label Studio.

Metrics: accuracy, macro F1, per-class F1, confusion matrix, training time and inference time per clip.

## Deliverables

- Validation and test metrics for every model.
- Annotation rules for the test videos.
- Correct and incorrect predictions, with a short note on the cause of each error.
- Screenshot of the interface running on a video that is not in the dataset.
- A justified recommendation of which model to use.

## Datasets

- Surf Maneuver Recognition (Hugging Face, Apache 2.0): https://huggingface.co/datasets/2nzi/surf-maneuvers
  - 1,062 clips in train/val/test splits, 4 classes: roller (523), cutback-frontside (238), 360 (180), take-off (121).
- SSD: Surfing skeleton dataset for temporal phase segmentation (IEEE DataPort, CC BY 4.0): https://doi.org/10.21227/w41m-k422

## Compute: NPAD/UFRN

- Training runs as SLURM jobs on the `gpu-4-a100` partition (`--gres=gpu:1`), never on the login node.
- Python 3.11 environment created with `uv` (the system Python is 3.6).
- Dataset and model cache stored under `/scratch/global/<user>/`, not in the home quota.
- Hugging Face weights are downloaded on the login node before submitting jobs.
- Job scripts live in `npad/` and log `hostname` and `nvidia-smi -L`.

## Repository structure

- app/ - Gradio web app
- data/ - local data, not versioned
- notebooks/ - exploratory analysis
- npad/ - SLURM job scripts for the NPAD supercomputer
- reports/ - figures and metrics
- src/data/ - dataset download, frame sampling and splits
- src/models/ - V-JEPA 2 probe and VideoMAE
- src/inference/ - surfer tracking, sliding window and timeline

## References

- Course material by Prof. Helton Maia: Deep Learning, ch. 4 (video classification with V-JEPA 2) and Computer Vision, ch. 5 (YOLO26).
- Assran et al. (2025). V-JEPA 2. Meta AI.
- Ultralytics YOLO26 (AGPL-3.0).

## Acknowledgments

This research was supported by NPAD/UFRN.
