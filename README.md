# Surf Maneuver Recognition

Recognition and temporal segmentation of surf maneuvers in video.

Final project for the Computer Vision course (PPGTI3007) at PPgTI/UFRN, 2026.2.

## Goal

Given a surf video, detect which maneuvers the surfer performs and when, producing a maneuver timeline (e.g. 0:12 take-off, 0:15 cutback, 0:19 roller).

## Approach

1. Data preparation: frame extraction, class balance analysis and splits.
2. Baseline: pretrained CNN applied per frame.
3. Main model: VideoMAE fine-tuned on the NPAD/UFRN supercomputer.
4. Long videos: surfer detection and tracking, sliding window and timeline.
5. Web app (Gradio): upload a video and get the detected maneuvers.

## Datasets

- Surf Maneuver Recognition (Hugging Face, Apache 2.0): https://huggingface.co/datasets/2nzi/surf-maneuvers
- SSD: Surfing skeleton dataset for temporal phase segmentation (IEEE DataPort, CC BY 4.0): https://doi.org/10.21227/w41m-k422

## Repository structure

- app/ - Gradio web app
- data/ - local data, not versioned
- notebooks/ - exploratory analysis
- npad/ - SLURM job scripts for the NPAD supercomputer
- reports/ - figures and metrics
- src/data/ - dataset download, frame extraction and splits
- src/models/ - CNN baseline and VideoMAE
- src/inference/ - tracking, sliding window and timeline

## Acknowledgments

This research was supported by NPAD/UFRN.
