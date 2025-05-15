export class ClassifyImage {
  /**
   * @param {ImageFile} imageFile
   * @param {Object} options
   * @param {tf.GraphModel|tf.LayersModel} options.model - loaded TensorFlow.js model
   * @param {string[]} options.labels - class labels for predictions
   */
  constructor(imageFile, { model, labels }) {
    this.imageFile = imageFile;
    this.model = model;
    this.labels = labels;
  }

  /**
   * Runs classification on the input imageFile
   * @returns {Promise<ImageFile>} - returns a new ImageFile with predictions added
   */
  async run() {
    if (!this.imageFile) {
      throw new Error('No ImageFile provided');
    }
    if (!this.model) {
      throw new Error('Model not loaded');
    }
    // Prepare tensor from image
    const tensor = await this.imageFile.toTensor({ targetWidth: 224, targetHeight: 224, normalize: true });

    // Run prediction
    const predictionTensor = this.model.predict(tensor);
    const predictions = await predictionTensor.data();

    // Map predictions to labels with scores
    const predictionArray = Array.from(predictions).map((score, i) => ({
      label: this.labels[i] || `class_${i}`,
      score,
    }));

    // Sort descending by score
    predictionArray.sort((a, b) => b.score - a.score);

    // Clean up
    tf.dispose([tensor, predictionTensor]);

    // Return new ImageFile with predictions attached
    return this.imageFile.cloneWithPredictions(predictionArray);
  }
}
