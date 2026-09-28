# PanDerm - Skin Lesion Segmentation

This implementation uses PanDerm for skin lesion segmentation.

## Install Environment

Main dependencies: Pytorch, Pytorch Lightning, MMSegmentation

```bash
cd segmentation
conda create -n dermseg python=3.10
conda activate dermseg
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

pip install -U openmim
mim install mmengine==0.10.4
mim install mmcv==2.1.0
mim install mmsegmentation==1.2.2
```

## Data Preparation

1. Download datasets:
   - [ISIC2018 task1](https://challenge.isic-archive.com/data/#2018)
   - [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. Change `--parent_path` in run.sh accordingly

## Fine-tuning with PanDerm Weight

1. **Download Pretrained Weights**
   - [PanDerm weights](https://drive.google.com/file/d/1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE/view)

2. **Set Pretrained Path**
   - Modify `--parent_path` in run.sh accordingly

3. **Start Training**
   ```bash
   bash run.sh
   ```
   Note: Adjust the path config to your desired storage location

4. **Evaluation**
   - Add `--evaluate` to `run.sh` for evaluation mode
   ```bash
   cd segmentation
   add --evaluate
   ```

## Starter Code for Your Own Dermoscopic Images

See [**evaluate.ipynb**](segmentation/evaluate.ipynb) to learn:
- Using it for skin lesion segmentation
