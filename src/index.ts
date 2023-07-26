import express, { Express, Request, Response } from 'express';
import fileUpload, { UploadedFile } from 'express-fileupload';
import sharp from 'sharp';
import { Rembg } from './rembg.js';

const app: Express = express(); 

app.use(fileUpload({
  safeFileNames: true,
  preserveExtension: true
}));

app.get('/', (req: Request, res: Response) => {
  res.send('Welcome to image background removal app');
})

app.post('/rembg', async (req: Request, res: Response) => {
  try {

    if (!req.files || Object.keys(req.files).length === 0) {
      return res.status(400).json({
        success: false,
        msg: 'No file was uploaded'
      });
    }

    const start = new Date();
    const file = req.files.sig as UploadedFile; //formdata name should be sig

    const img = sharp(file.data);

    const rembg = new Rembg({
      logging: true
    });

    const outputFile = await rembg.remove(img);

    if (outputFile == undefined) {
      throw new Error("Failed to process image");
    }

    const bufferImage = await outputFile.png().toBuffer();

    // encode to base64
    const base64String = Buffer.from(bufferImage).toString('base64');

    const timeDif = new Date().getTime() - start.getTime();
    const ms = timeDif / 1000;
    console.log(`took ${ms}ms to process`);

    return res.send({ success: true, payload: base64String });

  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, msg: 'Error processing image' });
  }
})

app.post('/remove',async (req: Request, res: Response) => {
  try {

    if (!req.files || Object.keys(req.files).length === 0) {
      return res.status(400).json({
        success: false,
        msg: 'No file was uploaded'
      });
    }

    const start = new Date();
    const file = req.files.sig as UploadedFile; //formdata name should be sig

    const img = sharp(file.data);

    const rembg = new Rembg({
      logging: true
    });

    const outputFile = await rembg.runInference(img);

    if (outputFile == undefined) {
      throw new Error("Failed to process image");
    }

    const bufferImage = await outputFile.png().toBuffer();

    // encode to base64
    const base64String = Buffer.from(bufferImage).toString('base64');

    const timeDif = new Date().getTime() - start.getTime();
    const ms = timeDif / 1000;
    console.log(`took ${ms}ms to process`);

    return res.send({ success: true, payload: base64String });

  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, msg: 'Error processing image' });
  }
})

const port = process.env['NODE_EXP_PORT'] ?? 5501;
app.listen(port, () => {
  console.log(`⚡[server]: Server started at http://localhost:${port}`)
})
