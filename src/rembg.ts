import * as fs from "fs/promises";
import pkg from "onnxruntime-node";
import * as os from "os";
import * as path from "path";
import sharp from "sharp";
import { gdown } from "./gdown.js";

const { InferenceSession, Tensor } = pkg;

function getMax(buffer: Buffer): number {
	let max = 0;
	for (let i = 0; i < buffer.length; i++) {
		if (buffer[i] > max) max = buffer[i];
	}
	return max;
}

function concatFloat32Array(arrays: Float32Array[]): Float32Array {
	let length = 0;
	for (const array of arrays) length += array.length;

	const output = new Float32Array(length);

	let outputIndex = 0;
	for (const array of arrays) {
		for (let n of array) {
			output[outputIndex] = n;
			outputIndex++;
		}
	}

	return output;
}

/**
 * checks if model exists on local device
 * @param path path to model
 * @returns 
 */
const exists = async (path: string) =>
	(await fs.stat(path).catch(() => {})) != null;

/**
 * Removal class
 */
export class Rembg {
  private modelDownloaded = false;
  private promisesResolvesUntillDownloaded: ((value: unknown) => void)[] = [];

  // the u2net model is a
  private readonly u2netHome = process.env["U2NET_HOME"] ?? path.resolve(os.homedir(), ".u2net");

  readonly modelPath = path.resolve(this.u2netHome, "u2net.onnx");
  readonly medModelPath = path.resolve(this.u2netHome, "u2net_med.onnx");

  private log(message?: any) {
    if (this.options.logging === false) return;
    console.log(message);
  }

  constructor(private readonly options: { logging?: boolean} = {}) {
    this.ensureModelDownloaded();
  }

  /**
   * Ensures download of the u2net model
   */
  private async ensureModelDownloaded() {
    if (await exists(this.modelPath)) {
      this.log("U2 model found");
      this.modelDownloaded = true;
    } else {
      this.log("U2-Net model downloading...");

      if (!(await exists(this.u2netHome))) await fs.mkdir(this.u2netHome);

      await gdown(
        "1tCU5MM1LhRgGou5OpmpjBQbSrYIUoYab",
        this.modelPath,
        "60024c5c889badc19c04ad937298a77b",
      );

      this.log("U2-Net model downloaded!");

      this.modelDownloaded = true;
    }

    for (const resolve of this.promisesResolvesUntillDownloaded) {
      resolve(null);
    }
  }

  /**
   * Converts buffer image to float32Array
   * @param buffer buffer of image
   * @param imageSize 
   * @returns 
   */
  private toFloat32Array(buffer: Buffer, imageSize: number) {
    const max = getMax(buffer);
    const mean = [0.485, 0.456, 0.406];
		const std = [0.229, 0.224, 0.225];

    // float32Array
    const inputChannels = [
      new Float32Array(imageSize * imageSize),
      new Float32Array(imageSize * imageSize),
      new Float32Array(imageSize * imageSize),
    ];

    for (let i = 0; i < buffer.length; i++) {
			const channel = i % 3;
			const channelIndex = Math.floor(i / 3);
			
			inputChannels[channel][channelIndex] =
				(buffer[i] / max - mean[channel]) / std[channel];
		}

    return inputChannels;
  }

  /**
   * Main method removes background
   * @param sharpInput image
   * @returns iamge with removed background
   */
  async remove(sharpInput: sharp.Sharp) {
    if (this.modelDownloaded == false) {
			await new Promise(resolve => {
				this.promisesResolvesUntillDownloaded.push(resolve);
			});
		}

    const imageSize = 320;
    const { width, height } = await sharpInput.metadata();

    if (width == undefined || height == undefined) {
      this.log(`Width or height is undefined. Width: ${width} & Height: ${height}`);
      return;
    }

    let bufferImage = await sharpInput.clone()
        .resize(imageSize, imageSize, { kernel: "lanczos3", fit: "fill" })
        .removeAlpha()
        .raw()
        .toBuffer();

    // float32Array
    const inputChannels = this.toFloat32Array(bufferImage, imageSize);

    const input = concatFloat32Array([
      inputChannels[2],
      inputChannels[0],
      inputChannels[1] 
    ]);

    const session = await InferenceSession.create(this.modelPath)
    const results = await session.run({
      "input.1": new Tensor("float32", input, [1, 3, imageSize, imageSize]),
    })

    const mostPreciseOutputName = String(
			Math.min(...session.outputNames.map((name: string | number) => +name)),
		);

		const outputMaskData = results[mostPreciseOutputName].data as Float32Array;

    for (let i = 0; i < outputMaskData.length; i++) {
      outputMaskData[i] = outputMaskData[i] * 255;
    }

    const sharpMask = await sharp(outputMaskData, {
        raw: { channels: 1, width: imageSize, height: imageSize },
      })
      .resize(width, height, { fit: 'fill' })
      .raw()
      .toBuffer();

    const finalPixels = await sharpInput.clone()
      .ensureAlpha()
      .raw()
      .toBuffer();

    for (let i = 0; i < finalPixels.length / 4; i++) {
      const alpha = sharpMask[i * 3];
      finalPixels[i * 4 + 3] = alpha;
    }
    
    // resize and return 
    return sharp(finalPixels, {
      raw: { channels: 4, width, height }
    });
  }

  /**
   * runs inference using a different model, u2net_med.onnx
   * not perfected yet
   */ 
  async runInference(image: sharp.Sharp) {
    const resolution = 1024;
    const { width, height } = await image.metadata();

    if (width == undefined || height == undefined) {
      this.log(`Width or height is undefined. Width: ${width} & Height: ${height}`);
      return;
    }

    const dims = [1, 3, resolution, resolution];

    // resize
    const resizedBuffer = await image.clone()
      .resize(resolution, resolution, { kernel: 'lanczos3', fit: 'fill'})
      .removeAlpha()
      .raw()
      .toBuffer();

    if (resizedBuffer == undefined) {
      throw new Error("Failed to resize image");
    }

    // convert to float32Array
    const mean = [128, 128, 128];
    const std = [255, 255, 255];
    const stride2 = resolution * resolution;
    const float32Data = new Float32Array(3 * stride2);

    for (let i = 0, j = 0; i < resizedBuffer.length; i += 4, j += 1) {
      float32Data[j] = (resizedBuffer[i] - mean[0]) / std[0];
      float32Data[j + stride2] = (resizedBuffer[i + 1] - mean[1]) / std[1];
      float32Data[j + stride2 + stride2] =
        (resizedBuffer[i + 2] - mean[2]) / std[2];
    }

    // get session
    const session = await InferenceSession.create(this.medModelPath)
    const results = await session.run({
      "input": new Tensor("float32", float32Data, dims),
    })

    const predictionDist = results['output'].data as Float32Array; // result -> output, 1890

    for (let i = 0; i < predictionDist.length; i ++) {
      predictionDist[i] = predictionDist[i] * 255;
    }

    const sharpMark = await sharp(predictionDist, {
      raw: { channels: 1, width: resolution, height: resolution, premultiplied: true }
    })
        .resize(width, height, { fit: 'fill' })
        .raw()
        .toBuffer();

    // recalculate proportions
    const finalPixel = await image.clone()
        .ensureAlpha()
        .raw({})
        .toBuffer();

    for (let i = 0; i < finalPixel.length / 4; i++) {
      let alpha = sharpMark[i * 3];
      finalPixel[i * 4 + 3] = alpha;
    }

    return sharp(finalPixel, {
      raw: { channels: 4, width, height }
    });
  }
}
