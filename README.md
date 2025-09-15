# Floor Plan Recovery using Palette Model

A PyTorch implementation of image-to-image diffusion models for floor plan recovery tasks. This project uses the Palette diffusion model architecture to reconstruct floor plans from conditional inputs.

## Experiment Results

All the experiment results can be downloaded from the link: [Experiment Results](https://drive.google.com/drive/folders/1pF2kcEQtI5ZRlZ_g0hJ_k6cxHfXubIxY?usp=drive_link)

## Project Structure

### Root Files

- **`run.py`** - Main training script for the Palette model. Contains the training loop, data loading, and model initialization for floor plan recovery tasks.
- **`eval.py`** - Evaluation script for testing trained models on test datasets.
- **`run_val.py`** - Validation script for model performance assessment.
- **`run_val_rw.py`** - Validation script for real-world dataset evaluation.
- **`run_abl_hm.py`** - Ablation study script for heatmap-related experiments.
- **`run_abl_traj.py`** - Ablation study script for trajectory-related experiments.
- **`run_val_abl_hm.py`** - Validation script for heatmap ablation studies.
- **`run_val_abl_traj.py`** - Validation script for trajectory ablation studies.

### Configuration (`config/`)

Contains JSON configuration files for different experimental setups:

- **`Palette_scalar100_sepe.json`** - Configuration for SE+PE experiments at scale 100
- **`Palette_scalar100_sexpe.json`** - Configuration for SExPE experiments at scale 100
- **`Palette_scalar100_abl_traj.json`** - Configuration for trajectory only ablation studies
- **`Palette_scalar100_sepe_abl_hm.json`** - Configuration for SE+PE heatmap only ablation studies
- **`Palette_scalar100_sexpe_abl_hm.json`** - Configuration for SExPE heatmap only ablation studies

### Data Management (`data/`)

Data loading and preprocessing modules:
- **`dataset.py`** - Main dataset classes, including `FloorPlanDataset` for handling floor plan images and conditions

### Datasets

#### `Dataset_Scale100_SEPE/`
Training and test data for SEPE experiments:
- **`train/`** - Training images with Condition_1, Condition_2, and Target folders
- **`test/`** - Test images with same structure
- **`Selected_50_train/`** - Curated subset of 50 training samples
- **`Selected_50_test/`** - Curated subset of 50 test samples
- **`id2idx.txt`** - Mapping file for image IDs to indices

#### `Dataset_Scale100_SExPE/`
Training and test data for SExPE experiments:
- Similar structure to SEPE dataset with train/test splits and selected subsets

#### `Dataset_rw/`
Real-world dataset with multiple experimental conditions:
- **`sepe_exp1_gmap/`** to **`sepe_exp5_gmap/`** - SEPE experiments 1-5 with ground truth maps
- **`sexpe_exp1_gmap/`** to **`sexpe_exp5_gmap/`** - SExPE experiments 1-5 with ground truth maps


## Usage

1. **Training**: Run `python run.py` with appropriate configuration file
2. **Validation**: Use `run_val*.py` scripts for validation on different datasets
3. **Evaluation**: Run `python eval.py` to evaluate trained models
3. **Ablation Studies**: Use `run_abl_*.py` scripts for ablation experiments

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies include PyTorch, TorchVision, OpenCV, TensorBoard, and other ML/CV libraries.

## Model Architecture

This project implements the Palette diffusion model for conditional image generation, specifically adapted for floor plan recovery tasks. The model uses a U-Net architecture with attention mechanisms for image-to-image translation.

## Reference

This implementation is based on the Palette model and adapted from:

**Implementation**: [Janspiry/Palette-Image-to-Image-Diffusion-Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)

**Original Paper**: Palette: Image-to-Image Diffusion Models  
*Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, Mohammad Norouzi*  
SIGGRAPH 2022  
[Paper](https://arxiv.org/abs/2111.05826)
