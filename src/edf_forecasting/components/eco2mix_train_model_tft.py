from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE
from sklearn.metrics import r2_score
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class Eco2mixTrainModelTFT:
    def __init__(self, params: dict):
        self.params = params
        self.model = None
        self.trainer = None

    def run(self, training, validation):
        # Model
        self.model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.params.get("learning_rate", 0.001),
            hidden_size=self.params.get("hidden_size", 128),
            attention_head_size=self.params.get("attention_head_size", 4),
            dropout=self.params.get("dropout", 0.1),
            hidden_continuous_size=self.params.get("hidden_continuous_size", 64),
            output_size=7,
            loss=QuantileLoss(quantiles=[0.025, 0.50, 0.975])
        )

        # Logger
        logger = TensorBoardLogger(
            save_dir=self.params.get("log_dir", "data/08_final_models/tft/logs"),
            name="lightning_logs_pipeline"
        )

        callbacks = []

        if self.params.get("use_early_stopping", False):
            callbacks.append(EarlyStopping(
                monitor=self.params.get("early_stopping_monitor", "val_loss"),
                patience=self.params.get("early_stopping_patience", 5),
                mode=self.params.get("early_stopping_mode", "min"),
                verbose=True
            ))

        if self.params.get("use_checkpoint", False):
            callbacks.append(ModelCheckpoint(
                dirpath=self.params.get("checkpoint_dir", "data/08_final_models/tft/checkpoints"),
                filename="epoch_{epoch:02d}-val_loss_{val_loss:.2f}",
                save_top_k=-1,
                monitor=self.params.get("early_stopping_monitor", "val_loss"),
                mode=self.params.get("early_stopping_mode", "min"),
                every_n_epochs=1
            ))

        self.trainer = Trainer(
            max_epochs=self.params.get("max_epochs", 20),
            log_every_n_steps=self.params.get("log_every_n_steps", 1),
            accelerator=self.params.get("accelerator", "cpu"),
            devices=self.params.get("devices", 1),
            precision=self.params.get("precision", "32-true"),
            gradient_clip_val=self.params.get("gradient_clip_val", 0.1),
            logger=logger,
            callbacks=callbacks
        )

        train_loader = training.to_dataloader(train=True, batch_size=self.params.get("batch_size", 64), num_workers=0)
        val_loader = validation.to_dataloader(train=False, batch_size=self.params.get("batch_size", 64) * 10, num_workers=0)
        
        print(f"Using accelerator: {self.trainer.accelerator}")
        self.trainer.fit(self.model, train_loader, val_loader)

        # Evaluate
        raw_predictions = self.model.predict(val_loader, mode="raw", return_x=True)
        pred_2_5 = raw_predictions.output.prediction[:, :, 0]
        pred_97_5 = raw_predictions.output.prediction[:, :, 2]
        true_values = raw_predictions.x["decoder_target"]

        mae = MAE()(pred_97_5, true_values)
        rmse = RMSE()(pred_97_5, true_values)
        r2 = r2_score(true_values.cpu().numpy().flatten(), pred_97_5.cpu().numpy().flatten())
        coverage = ((true_values >= pred_2_5) & (true_values <= pred_97_5)).float().mean().item() * 100

        training_info = {
            "model": "TemporalFusionTransformer",
            "epochs": self.trainer.current_epoch,
            "seed": self.params.get("seed", "not_specified")
        }

        scores = {
            "MAE": float(mae.item()),
            "RMSE": float(rmse.item()),
            "R2": float(r2),
            "Coverage": float(coverage)
        }

        return self.model, training_info, scores
