/**
 * This file contains methods to process data. The 2 main categories of data processing tasks include
 * - preparing the input for inference/training task
 * - encoding the raw images output into image 
 * At this time we are only dealing with image data. This includes single image, image stack and videos. All of these requires separate processing 
 * 
 */
import * as tf from '@tensorflow/tfjs';

// export const process_input = async (files,options={result:'raw_image'})=>{
//   if (!files || files.length === 0) {
//     throw new Error('No input files provided.');
//   }
//   const allowedExtensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.mp4', '.webm'];
//   const allowedMimeTypes = ['image/png', 'image/jpeg', 'image/bmp', 'image/gif', 'video/mp4', 'video/webm'];

//   const fileArray = Array.from(files);
//   const output = [];

//   for (const file of fileArray) {
//     const extension = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
//     // Check extension and MIME type
//     if (!allowedExtensions.includes(extension) || !allowedMimeTypes.includes(file.type)) {
//       console.warn(`Skipping unsupported file: ${file.name}`);
//       continue;
//     }
//     const buffer = await file.arrayBuffer();
//     output.push({
//       file_raw: buffer,
//       file_type: file.type,
//       file_name: file.name,
//       file_extension: extension
//     });
//   }

//   if (output.length === 0) {
//     throw new Error('No valid files were provided.');
//   }
//   return output
// }


// export class ImageFile {
//   constructor({
//     file,
//     file_raw,
//     file_type,
//     file_name,
//     file_extension,
//     image_width,
//     image_height,
//   }) {
//     this.file = file;
//     this.file_raw = file_raw;
//     this.file_type = file_type;
//     this.file_name = file_name;
//     this.file_extension = file_extension;
//     this.image_width = image_width;
//     this.image_height = image_height;
//     this._cachedImage = null;
//   }

//   /**
//    * Factory method to create an ImageFile instance from a File object
//    * @param {File} file
//    * @returns {Promise<ImageFile>}
//    */
//   static async fromFile(file) {
//     if (!(file instanceof File)) {
//       throw new Error('Expected a File object.');
//     }

//     if (!file.type.startsWith('image/')) {
//       throw new Error(`Invalid file type: ${file.type}`);
//     }

//     const file_raw = await ImageFile._readFileAsArrayBuffer(file);
//     const file_name = file.name;
//     const file_type = file.type;
//     const file_extension = ImageFile._extractExtension(file_name);

//     // Load the image to get dimensions
//     const blob = new Blob([file_raw], { type: file_type });
//     const url = URL.createObjectURL(blob);
//     const img = await ImageFile._loadImageFromURL(url);
//     URL.revokeObjectURL(url);

//     return new ImageFile({
//       file,
//       file_raw,
//       file_type,
//       file_name,
//       file_extension,
//       image_width: img.naturalWidth,
//       image_height: img.naturalHeight,
//     });
//   }

//   /**
//    * Returns a tensor suitable for TensorFlow input: [1, H, W, 3]
//    * @param {Object} options
//    * @param {number} options.targetWidth
//    * @param {number} options.targetHeight
//    * @param {boolean} options.normalize
//    * @returns {Promise<tf.Tensor4D>}
//    */
//   async toTensor({ targetWidth = 256, targetHeight = 256, normalize = true } = {}) {
//     const img = await this._getHTMLImage();
//     return tf.tidy(() => {
//       let input = tf.browser.fromPixels(img);
//       input = tf.image.resizeBilinear(input, [targetHeight, targetWidth]);
//       if (normalize) {
//         input = tf.div(input, 255);
//       }
//       return input.expandDims(0); // [1, H, W, 3]
//     });
//   }

//   /**
//    * Internal: Returns cached or new HTMLImageElement
//    * @returns {Promise<HTMLImageElement>}
//    */
//   async _getHTMLImage() {
//     if (this._cachedImage) return this._cachedImage;

//     const blob = new Blob([this.file_raw], { type: this.file_type });
//     const url = URL.createObjectURL(blob);
//     const img = await ImageFile._loadImageFromURL(url);
//     URL.revokeObjectURL(url);
//     this._cachedImage = img;
//     return img;
//   }

//   static _extractExtension(name) {
//     const parts = name.split('.');
//     return parts.length > 1 ? parts.pop().toLowerCase() : '';
//   }

//   static _readFileAsArrayBuffer(file) {
//     return new Promise((resolve, reject) => {
//       const reader = new FileReader();
//       reader.onload = () => resolve(reader.result);
//       reader.onerror = reject;
//       reader.readAsArrayBuffer(file);
//     });
//   }

//   static _loadImageFromURL(url) {
//     return new Promise((resolve, reject) => {
//       const img = new Image();
//       img.onload = () => resolve(img);
//       img.onerror = reject;
//       img.src = url;
//     });
//   }
// }


// a general image Object 
export class ImageFile {
  /**
   * @param {Object} args
   * @param {File} args.file
   * @param {ArrayBuffer} args.file_raw
   * @param {string} args.file_type
   * @param {string} args.file_name
   * @param {string} args.file_extension
   * @param {number} args.image_width
   * @param {number} args.image_height
   * @param {Array<{ label: string, score: number }>} [args.predictions]
   */
  constructor({
    file,
    file_raw,
    file_type,
    file_name,
    file_extension,
    image_width,
    image_height,
    predictions = null,
  }) {
    if (!file_type.startsWith('image/')) {
      throw new Error(`Invalid file type for ImageFile: ${file_type}`);
    }

    this.file = file;
    this.file_raw = file_raw;
    this.file_type = file_type;
    this.file_name = file_name;
    this.file_extension = file_extension;
    this.image_width = image_width;
    this.image_height = image_height;
    this.predictions = predictions;

    this._blobURL = null;
    this._cachedTensor = null;
  }

  static async fromFile(file) {
    const file_raw = await file.arrayBuffer();
    const file_type = file.type;
    const file_name = file.name;
    const file_extension = file.name.split('.').pop().toLowerCase();

    const blob = new Blob([file_raw], { type: file_type });
    const url = URL.createObjectURL(blob);
    const img = await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        URL.revokeObjectURL(url);
        resolve(image);
      };
      image.onerror = reject;
      image.src = url;
    });

    return new ImageFile({
      file,
      file_raw,
      file_type,
      file_name,
      file_extension,
      image_width: img.naturalWidth,
      image_height: img.naturalHeight,
    });
  }

  toBlob() {
    return new Blob([this.file_raw], { type: this.file_type });
  }

  toObjectURL() {
    if (!this._blobURL) {
      this._blobURL = URL.createObjectURL(this.toBlob());
    }
    return this._blobURL;
  }

  async toTensor({ targetWidth = 224, targetHeight = 224, normalize = true } = {}) {
    const img = await this._loadHTMLImage();
    return tf.tidy(() => {
      let tensor = tf.browser.fromPixels(img);
      tensor = tf.image.resizeBilinear(tensor, [targetHeight, targetWidth]);
      if (normalize) {
        tensor = tf.div(tensor, 255.0);
      }
      return tensor.expandDims(0); // [1, H, W, 3]
    });
  }

  _loadHTMLImage() {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = this.toObjectURL();
    });
  }

  cloneWithPredictions(predictions) {
    return new ImageFile({
      ...this,
      predictions,
    });
  }

  topPredictions(n = 1) {
    return this.predictions ? this.predictions.slice(0, n) : [];
  }

  bestPrediction() {
    return this.predictions?.[0] || null;
  }
}


export class VideoFile {
  /**
   * @param {File} file - Input video file
   * @param {Array<ImageFile>} frames - Extracted frames
   * @param {number} fps - FPS used for extraction
   */
  constructor(file, frames, fps) {
    this.file = file;
    this.frames = frames;
    this.fps = fps;
  }

  /**
   * Extract frames from a video file at given FPS
   * @param {File} file
   * @param {number} fps
   * @returns {Promise<VideoFile>}
   */
  static async fromFile(file, fps = 1) {
    const videoBlobUrl = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.src = videoBlobUrl;
    video.crossOrigin = 'anonymous';
    video.muted = true;

    await video.play();
    await new Promise((res) => video.onloadedmetadata = res);
    video.pause();

    const duration = video.duration;
    const totalFrames = Math.floor(duration * fps);
    const frames = [];

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    for (let i = 0; i < totalFrames; i++) {
      const time = i / fps;
      await seekVideo(video, time);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise(res => canvas.toBlob(res, 'image/png'));
      const file_raw = await blob.arrayBuffer();

      const frame = new ImageFile({
        file: null,
        file_raw,
        file_type: 'image/png',
        file_name: `${file.name}_frame_${i}.png`,
        file_extension: 'png',
        image_width: canvas.width,
        image_height: canvas.height,
      });

      frames.push(frame);
    }

    URL.revokeObjectURL(videoBlobUrl);
    return new VideoFile(file, frames, fps);
  }

  /**
   * Run model on every frame
   * @param {(img: ImageFile) => Promise<ImageFile>} modelFn
   * @returns {Promise<Array<ImageFile>>}
   */
  async runOnEach(modelFn) {
    return Promise.all(this.frames.map(f => modelFn(f)));
  }
}

// Helper to seek video
function seekVideo(video, time) {
  return new Promise((resolve) => {
    const listener = () => {
      video.removeEventListener('seeked', listener);
      resolve();
    };
    video.addEventListener('seeked', listener);
    video.currentTime = time;
  });
}
