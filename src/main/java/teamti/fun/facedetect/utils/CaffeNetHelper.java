package teamti.fun.facedetect.utils;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static teamti.fun.facedetect.utils.ResourceHelper.copyResourceToTempFile;

public class CaffeNetHelper {

    private static final Net net;
    private static final String PROTO_FILE;
    private static final String CAFFE_MODEL_FILE;

    static {
        try {
            PROTO_FILE = copyResourceToTempFile("models/deploy.prototxt").toString();
            CAFFE_MODEL_FILE = copyResourceToTempFile("models/res10_300x300_ssd_iter_140000.caffemodel").toString();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        net = readNetFromCaffe(PROTO_FILE, CAFFE_MODEL_FILE);
    }

    public static List<double[]> detectAndDrawWithSmoothTransition(Mat image, GraphicsContext gc, double canvasWidth, double canvasHeight, List<double[]> prevCoordsList, boolean firstFrame) {
        int originalWidth = image.cols();
        int originalHeight = image.rows();

        // Create a 4-dimensional blob from image with NCHW (Number of images in the batch, Channel, Height, Width) dimensions order
        Mat blob = blobFromImage(image, 1.0, new Size(300, 300), new Scalar(104.0, 177.0, 123.0, 0), false, false, CV_32F);

        // Set the input to network model
        net.setInput(blob);

        // Feed forward the input to the network to get the output matrix
        Mat output = net.forward();

        // Extract a 2D matrix for 4D output matrix with form of (number of detections x 7)
        Mat detections = new Mat(new Size(output.size(3), output.size(2)), CV_32F, output.ptr(0, 0));

        // Create indexer to access elements of the matrix
        FloatIndexer srcIndexer = detections.createIndexer();

        // Clear the canvas
        gc.clearRect(0, 0, canvasWidth, canvasHeight);

        double xScale = canvasWidth / originalWidth;
        double yScale = canvasHeight / originalHeight;

        List<double[]> newCoordsList = new ArrayList<>();
        List<double[]> smoothCoordsList = new ArrayList<>();

        for (int i = 0; i < detections.size(0); i++) {
            float confidence = srcIndexer.get(i, 2);
            float f1 = srcIndexer.get(i, 3);
            float f2 = srcIndexer.get(i, 4);
            float f3 = srcIndexer.get(i, 5);
            float f4 = srcIndexer.get(i, 6);
            if (confidence > 0.55) {
                int x1 = Math.round(f1 * originalWidth);
                int y1 = Math.round(f2 * originalHeight);
                int x2 = Math.round(f3 * originalWidth);
                int y2 = Math.round(f4 * originalHeight);

                // Apply scaling to coordinates
                double x1Scaled = x1 * xScale;
                double y1Scaled = y1 * yScale;
                double x2Scaled = x2 * xScale;
                double y2Scaled = y2 * yScale;

                double[] newCoords = new double[]{x1Scaled, y1Scaled, x2Scaled, y2Scaled};
                newCoordsList.add(newCoords);

                if (firstFrame || prevCoordsList.size() <= i) {
                    prevCoordsList.add(newCoords.clone());
                }

                double[] prevCoords = prevCoordsList.get(i);

                // Smooth transition using linear interpolation
                double smoothedX1 = (prevCoords[0] + newCoords[0]) / 2;
                double smoothedY1 = (prevCoords[1] + newCoords[1]) / 2;
                double smoothedX2 = (prevCoords[2] + newCoords[2]) / 2;
                double smoothedY2 = (prevCoords[3] + newCoords[3]) / 2;

                smoothCoordsList.add(new double[]{smoothedX1, smoothedY1, smoothedX2, smoothedY2});

                // Draw the rectangle around the detected face
                gc.setStroke(Color.GREEN);
                gc.setLineWidth(3);
                gc.strokeRect(smoothedX1, smoothedY1, smoothedX2 - smoothedX1, smoothedY2 - smoothedY1);

                // Prepare the text with the confidence level
                String label = String.format("Độ chính xác: %d%%", Math.round(confidence * 100));

                // Set the font size and thickness
                gc.setFont(new javafx.scene.text.Font("Arial", 18)); // Increase font size
                gc.setLineWidth(5); // Increase text thickness

                // Draw the text above the rectangle
                gc.setFill(Color.WHITE);
                gc.fillText(label, smoothedX1, smoothedY1 > 10 ? smoothedY1 - 10 : smoothedY1 + 20); // Adjusted y-coordinate to account for increased font size

                // Update previous coordinates
                prevCoordsList.set(i, newCoords);
            }
        }

        // Remove extra previous coordinates if the number of faces decreased
        while (prevCoordsList.size() > newCoordsList.size()) {
            prevCoordsList.removeLast();
        }

        return smoothCoordsList;
    }
}
