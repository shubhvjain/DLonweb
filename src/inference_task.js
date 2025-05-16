
// import * as cocoSsd from '@tensorflow-models/coco-ssd';
// import '@tensorflow/tfjs'; // make sure TFJS is loaded

/**
 * Load TensorFlow.js model either from:
 * - a library folder path (URL string)
 * - or user uploaded files (jsonFile + binFiles array)
 *
 * @param {Object} params
 * @param {string} [params.libraryPath] - URL to model.json in your library folder
 * @param {File} [params.userJsonFile] - user-uploaded model.json file
 * @param {File[]} [params.userWeightFiles] - user-uploaded binary weight files
 * @returns {Promise<tf.GraphModel|tf.LayersModel>}
 */
import {ImageFile} from "./data"
export async function loadModel({ libraryPath, userJsonFile, userWeightFiles }) {
  if (libraryPath) {
    // Load from library folder URL
    return tf.loadGraphModel(libraryPath);
  }

  if (userJsonFile && userWeightFiles?.length) {
    // Load from user files
    const model = await tf.loadGraphModel(
      tf.io.browserFiles([userJsonFile, ...userWeightFiles])
    );
    return model;
  }

  throw new Error('Insufficient model files provided.');
}



export class InferenceTask {
  /**
   * @param {'classify'|'segment'} task_type
   * @param {ImageFile|VideoFile|ImageStack} input
   * @param {tf.GraphModel|tf.LayersModel} model
   * @param {Object} options - additional options like labels, targetWidth, targetHeight, etc.
   */
  constructor(task_type, input, model=null, options = {}) {
    this.task_type = task_type;
    this.input = input;
    this.model = model;
    this.options = options;
  }

  /**
   * Run the inference task, returns same input type but with predictions attached
   * @returns {Promise<ImageFile|VideoFile|ImageStack>}
   */
  async run() {
    switch (this.task_type) {
      case 'classify':
        return this._runGeneralClassification();
      case 'segment':
        return this._runSegmentation();
      default:
        throw new Error(`Unsupported task type: ${this.task_type}`);
    }
  }
  async get_output(){
    // run the inference task and generate the output 
    let result = await this.run()
    console.log(result)
  }


  async _runSegmentation() {
    const targetWidth = this.options.targetWidth || 224;
    const targetHeight = this.options.targetHeight || 224;

    // Helper: segment single ImageFile
    const segmentImage = async (imageFile) => {
      const tensor = await imageFile.toTensor({ targetWidth, targetHeight, normalize: true });
      const outputTensor = this.model.predict(tensor);
      // Assuming model output is segmentation map: [1, H, W, C]

      // Post-processing can vary — here just attach raw tensor data as predictions
      const segmentationData = await outputTensor.array();

      tf.dispose([tensor, outputTensor]);

      return imageFile.cloneWithPredictions(segmentationData);
    };

    if (this.input instanceof ImageFile) {
      return segmentImage(this.input);
    }

    if (this.input instanceof VideoFile || this.input instanceof ImageStack) {
      const frames = await Promise.all(this.input.frames.map(segmentImage));
      return new this.input.constructor(this.input.file, frames, this.input.fps || null);
    }

    throw new Error('Unsupported input type for segmentation');
  }
    // New GeneralClassification using COCO-SSD model
    async _runGeneralClassification() {
      // If model not loaded, load coco-ssd model internally
      if (!this.model) {
        this.model = await cocoSsd.load();
      }
  
      // Helper: run detection on single ImageFile
      const detectImage = async (imageFile) => {
        const imgEl = await imageFile._loadHTMLImage();
        const predictions = await this.model.detect(imgEl);
        // Attach predictions to ImageFile (you may need to adjust cloneWithPredictions)
        return imageFile.cloneWithPredictions(predictions);
      };
  
      if (this.input instanceof ImageFile) {
        return detectImage(this.input);
      }
  
      if (this.input instanceof VideoFile || this.input instanceof ImageStack) {
        const frames = await Promise.all(this.input.frames.map(detectImage));
        return new this.input.constructor(this.input.file, frames, this.input.fps || null);
      }
  
      throw new Error('Unsupported input type for GeneralClassification');
  }
}
