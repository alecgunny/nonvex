import numpy as np
import tensorflow as tf
from hermes.typeo import typeo


@typeo
def main(learning_rate: float, batch_size: int, hidden_dim: int):
    model = tf.keras.Sequential(
        layers=[
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate)
    model.compile(optimizer, "mse")

    X = 0.1 * np.random.randn(1000, 10)
    y = np.random.randint(2, size=(1000, 1))
    X[y[:, 0] == 1] += 0.05

    history = model.fit(
        X, y, validation_split=0.25, batch_size=batch_size, epochs=5
    ).history
    return {"val_loss": min(history["val_loss"])}


if __name__ == "__main__":
    main()
