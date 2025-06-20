package com.example;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.nio.FloatBuffer;
import java.util.Collections;

import static java.lang.Math.*;

public class Main {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static float normalizeRadians(float angle) {
        return (float) (angle - 2 * Math.PI * Math.floor((angle + Math.PI) / (2 * Math.PI)));
    }

    public static void main(String[] args) {
        String modelPath = "src/main/resources/model/palm_detection_full_inf_post_192x192.onnx";  // ✅ use HandLandmarkDetector
        String imagePath = "src/main/resources/img(1).jpg";

        try {
            // Load and preprocess image
            //Mat originalImage = Imgcodecs.imread(imagePath);
//            if (originalImage.empty()) {
//                System.out.println("❌ Image not found: " + imagePath);
//                return;
//            }
            VideoCapture cap = new VideoCapture("src/main/resources/hand.mp4");
            if (!cap.isOpened()) {
                System.out.println("Error: Cannot open video file.");
                return;
            }

            // Get video properties
            double fps = cap.get(Videoio.CAP_PROP_FPS);
            Size frameSize = new Size(640, 480);

            // Define the codec and create VideoWriter object
            int fourcc = VideoWriter.fourcc('M', 'J', 'P', 'G'); // use 'X', 'V', 'I', 'D' for .avi
            VideoWriter writer = new VideoWriter("src/main/resources/out.avi", fourcc, fps, frameSize, true);

            if (!writer.isOpened()) {
                System.out.println("Error: Cannot open video writer.");
                cap.release();
                return;
            }

            Mat originalImage = new Mat();
            int inputWidth = 192;
            int inputHeight = 192;
            while (cap.read(originalImage)) {
                if(originalImage.empty())
                    break;
                long startTime = System.nanoTime();
                Imgproc.resize(originalImage, originalImage, new Size(640, 480));
                Mat resized = new Mat();
                Imgproc.resize(originalImage, resized, new Size(inputWidth, inputHeight));
                Imgproc.cvtColor(resized, resized, Imgproc.COLOR_BGR2RGB);
                resized.convertTo(resized, CvType.CV_32FC3, 1.0 / 255);

                // HWC to CHW
                float[] inputData = new float[3 * inputHeight * inputWidth];
                int idx = 0;
                for (int c = 0; c < 3; c++) {
                    for (int y = 0; y < inputHeight; y++) {
                        for (int x = 0; x < inputWidth; x++) {
                            double[] pixel = resized.get(y, x);
                            inputData[idx++] = (float) pixel[c];
                        }
                    }
                }

                // Load model and run inference
                try (OrtEnvironment env = OrtEnvironment.getEnvironment();
                     OrtSession session = env.createSession(modelPath, new OrtSession.SessionOptions())) {

                    OnnxTensor inputTensor = OnnxTensor.createTensor(
                            env, FloatBuffer.wrap(inputData), new long[]{1, 3, 192, 192});

                    OrtSession.Result result = session.run(Collections.singletonMap("input", inputTensor)); // ✅ Input name may vary

                    // Get landmarks output (expected shape: [1, 63])
                    float[][] boxes = (float[][]) result.get(0).getValue();
                    for (float[] box : boxes) {
                        float pdScore = box[0];
                        float boxX = box[1];
                        float boxY = box[2];
                        float boxSize = box[3];
                        float kp0X = box[4];
                        float kp0Y = box[5];
                        float kp2X = box[6];
                        float kp2Y = box[7];
                        if (pdScore < 0.8)
                            break;
                        if (boxSize > 0) {
                            float kp02X = kp2X - kp0X;
                            float kp02Y = kp2Y - kp0Y;
                            float sqnRrSize = 2.9f * boxSize;
                            float rotation = (float) (0.5 * Math.PI - Math.atan2(-kp02Y, kp02X));
                            rotation = normalizeRadians(rotation);

                            float sqnRrCenterX = (float) (boxX + 0.5 * boxSize * Math.sin(rotation));
                            float sqnRrCenterY = (float) (boxY - 0.5 * boxSize * Math.cos(rotation));
                            int square_standard_size = max(originalImage.width(), originalImage.height());
                            int square_padding_half_size = abs(originalImage.height() - originalImage.width()) / 2;
                            sqnRrCenterY = (sqnRrCenterY * square_standard_size - square_padding_half_size) / originalImage.height();

                            float centerX = sqnRrCenterX * originalImage.width();
                            float centerY = sqnRrCenterY * originalImage.height();
                            float size = sqnRrSize * originalImage.width(); // assuming square in image coords
                            float halfSize = size / 2;
//                        double cosR = Math.cos(rotation);
//                        double sinR = Math.sin(rotation);
//                        Point[] corners = new Point[4];
//                        corners[0] = new Point(centerX - halfSize * cosR + halfSize * sinR, centerY - halfSize * sinR - halfSize * cosR); // top-left
//                        corners[1] = new Point(centerX + halfSize * cosR + halfSize * sinR, centerY + halfSize * sinR - halfSize * cosR); // top-right
//                        corners[2] = new Point(centerX + halfSize * cosR - halfSize * sinR, centerY + halfSize * sinR + halfSize * cosR); // bottom-right
//                        corners[3] = new Point(centerX - halfSize * cosR - halfSize * sinR, centerY - halfSize * sinR + halfSize * cosR); // bottom-left
//                        Point[] corners = new Point[4];
//                        corners[0] = new Point(centerX - halfSize, centerY - halfSize); // top-left
//                        corners[1] = new Point(centerX + halfSize, centerY - halfSize); // top-right
//                        corners[2] = new Point(centerX + halfSize, centerY + halfSize); // bottom-right
//                        corners[3] = new Point(centerX - halfSize, centerY + halfSize); // bottom-left
//                        for (int j = 0; j < 4; j++) {
//                            Imgproc.line(originalImage, corners[j], corners[(j + 1) % 4], new Scalar(0, 255, 0), 2);
//                        }
                            int x = max(0, (int) (centerX - halfSize));
                            int y = max(0, (int) (centerY - halfSize));
                            int w = min(originalImage.width(), (int) (halfSize * 2) + x) - x;
                            int h = min(originalImage.height(), (int) (halfSize * 2) + y) - y;
                            Rect roi = new Rect(x, y, w, h);
                            Mat cropped = new Mat(originalImage, roi);
                            get_Landmarks(originalImage, cropped, x, y);
                        }
                    }
                    //Imgcodecs.imwrite(outputPath, originalImage); // Save output
                }
                // [Your frame processing code here]

                long endTime = System.nanoTime();
                double elapsedMs = (endTime - startTime) / 1_000_000.0;
                String timeString = String.format("Time: %.2f ms", elapsedMs);
                Imgproc.putText(
                        originalImage,
                        timeString,
                        new Point(10, 30),                  // x=10, y=30
                        Imgproc.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        new Scalar(255, 255, 255),          // white color (BGR)
                        2
                );
                writer.write(originalImage);
            }
            cap.release();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void get_Landmarks(Mat originalImage, Mat hand, int left, int top) {
        String modelPath = "src/main/resources/model/hand_landmark.onnx";
        String outputPath = "src/main/resources/points_output.png";
        int inputWidth = 256;
        int inputHeight = 256;
        Mat resized = new Mat();

        Imgproc.resize(hand, resized, new Size(inputWidth, inputHeight));
        Imgproc.cvtColor(resized, resized, Imgproc.COLOR_BGR2RGB);
        resized.convertTo(resized, CvType.CV_32FC3, 1.0 / 255);
        float[] inputData = new float[3 * inputHeight * inputWidth];
        int idx = 0;
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < inputHeight; y++) {
                for (int x = 0; x < inputWidth; x++) {
                    double[] pixel = resized.get(y, x);
                    inputData[idx++] = (float) pixel[c];
                }
            }
        }

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             OrtSession session = env.createSession(modelPath, new OrtSession.SessionOptions())) {

            OnnxTensor inputTensor = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(inputData), new long[]{1, 3, inputWidth, inputHeight});

            OrtSession.Result result = session.run(Collections.singletonMap("image", inputTensor)); // ✅ Input name may vary
            float[] scores = (float[]) result.get(0).getValue();
            float[][][] landmarks = (float[][][]) result.get(2).getValue();
            for (int i = 0; i < landmarks[0].length; i++) {
                float px = left + landmarks[0][i][0] * hand.width();
                float py = top + landmarks[0][i][1] * hand.height();
                Imgproc.circle(originalImage, new Point(px, py), 3, new Scalar(255, 0, 0), -1);
            }
            //Imgcodecs.imwrite(outputPath, originalImage); // Save output

        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }
}
