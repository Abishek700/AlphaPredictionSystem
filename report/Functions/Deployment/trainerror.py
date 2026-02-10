
    def train_model(features, labels):
        try:
            model = createCnnModel()
            model.fit(features, labels, epochs=10, batch_size=32, validation_split=0.2)
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise
    \end{lstlisting}