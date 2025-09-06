import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import baseline
from SingularTrajectory import *
from utils import *

# -----------------------------------------------------------------------------
# ログ設定のための簡易的な設定
# -----------------------------------------------------------------------------
import logging
# --- NEW: タイムスタンプとファイルコピーのためにインポート ---
from datetime import datetime
import shutil
# --- FIX: argparseをインポート ---
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="./config/singulartrajectory-transformerdiffusion-zara1.json", type=str, help="config file path")
    parser.add_argument('--tag', default="SingularTrajectory-TEMP", type=str, help="personal tag for the model")
    parser.add_argument('--device', default="gpu", type=str, choices=["cpu", "gpu"], help="device for the model")
    parser.add_argument('--gpu_id', default="0", type=str, help="gpu id for the model")
    parser.add_argument('--test', default=False, action='store_true', help="evaluation mode")
    args = parser.parse_args()

    logger.info("===== Arguments =====")
    print_arguments(vars(args))

    logger.info("===== Configs =====")
    hyper_params = get_exp_config(args.cfg)
    print_arguments(hyper_params)

    if args.device == "gpu" and torch.cuda.is_available() and args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    PredictorModel = getattr(baseline, hyper_params.baseline).TrajectoryPredictor
    hook_func = DotDict({"model_forward_pre_hook": getattr(baseline, hyper_params.baseline).model_forward_pre_hook,
                         "model_forward": getattr(baseline, hyper_params.baseline).model_forward,
                         "model_forward_post_hook": getattr(baseline, hyper_params.baseline).model_forward_post_hook})
    ModelTrainer = getattr(trainer, *[s for s in trainer.__dict__.keys() if hyper_params.baseline in s.lower()])
    trainer = ModelTrainer(base_model=PredictorModel, model=SingularTrajectory, hook_func=hook_func,
                           args=args, hyper_params=hyper_params, device=args.device)

    # --- MODIFIED: Override the checkpoint directory to save/load from ./model/ ---
    # This ensures models are saved in a consistent, user-specified location.
    save_directory = os.path.join('.', 'model')
    trainer.checkpoint_dir = save_directory
    logger.info(f"モデルの保存/読み込み先を '{save_directory}' に変更しました。")
    
    # Ensure the directory exists before training starts
    if not args.test and not os.path.exists(save_directory):
        os.makedirs(save_directory)
        logger.info(f"モデル保存ディレクトリを作成しました: {save_directory}")


    if not args.test:
        trainer.init_descriptor()
        trainer.fit()

        # --- NEW: Create a timestamped backup of the best model after training ---
        logger.info("バックアップを作成しています...")
        best_model_path = os.path.join(save_directory, 'model_best.pth')
        if os.path.exists(best_model_path):
            try:
                now = datetime.now()
                timestamped_filename = f'model_best_{now.strftime("%Y%m%d_%H%M%S")}.pth'
                backup_path = os.path.join(save_directory, timestamped_filename)
                shutil.copy(best_model_path, backup_path)
                logger.info(f"✅ ベストモデルのバックアップを '{backup_path}' に作成しました。")
            except Exception as e:
                logger.warning(f"⚠️ モデルのバックアップ作成に失敗しました: {e}")
        else:
            logger.warning("⚠️ ベストモデルファイルが見つからなかったため、バックアップは作成されませんでした。")

    else:
        # --- MODIFIED: load_model will now use the overridden path ---
        trainer.load_model()
        logger.info("Testing...")
        results = trainer.test()
        logger.info(f"Scene: {hyper_params.dataset} " + " ".join([f"{meter}: {value:.4f}" for meter, value in results.items()]))


