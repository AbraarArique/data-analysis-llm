import torch
from chronos import ChronosPipeline
import matplotlib.pyplot as plt
import numpy as np

# Set up Chronos T5 for inference
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)


# Generate forecasts and create visual graph
def chronos_prediction(historical, length):
    context = torch.tensor(historical)
    forecast = pipeline.predict(context, length)

    forecast_index = range(len(historical), len(historical) + length)
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(historical, color="blue", label="Historical")
    plt.plot(forecast_index, median, color="green", label="Median Forecast")
    plt.fill_between(
        forecast_index,
        low,
        high,
        color="green",
        alpha=0.3,
        label="80% prediction interval",
    )
    plt.legend()
    plt.grid()
    plt.show()

    return median
