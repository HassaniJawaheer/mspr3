import pickle
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, RMSE, QuantileLoss
from sklearn.metrics import r2_score

class Eco2mixEvaluateModelTFT:
    def __init__(self, batch_size: int = 256):
        self.batch_size = batch_size

    def run(self, model_path: str, test_dataset):
        model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        test_loader = test_dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)

        raw_predictions = model.predict(test_loader, mode="raw", return_x=True)
        pred_median = raw_predictions.output.prediction[:, :, 3]
        true_values = raw_predictions.x["decoder_target"]

        mae = MAE()(pred_median, true_values)
        rmse = RMSE()(pred_median, true_values)
        r2 = r2_score(true_values.cpu().numpy().flatten(), pred_median.cpu().numpy().flatten())
        pred_5 = raw_predictions.output.prediction[:, :, 0]
        pred_95 = raw_predictions.output.prediction[:, :, 6]
        coverage = ((true_values >= pred_5) & (true_values <= pred_95)).float().mean().item() * 100
        quantile_loss = QuantileLoss()(raw_predictions.output.prediction, true_values)

        return {
            "MAE": float(mae.item()),
            "RMSE": float(rmse.item()),
            "R2": float(r2),
            "Coverage": float(coverage),
            "QuantileLoss": float(quantile_loss.item())
        }
