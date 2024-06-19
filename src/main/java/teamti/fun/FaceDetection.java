package teamti.fun;

import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.*;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class FaceDetection {

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

    public static void detectAndDraw(Mat image) {

        // Get the original image size
        int originalWidth = image.size().width();
        int originalHeight = image.size().height();

        // Create a 4-dimensional blob from image with NCHW dimensions order
        Mat blob = blobFromImage(image, 1.0, new Size(300, 300), new Scalar(104.0, 177.0, 123.0, 0), false, false, CV_32F);

        // Set the input to network model
        net.setInput(blob);

        // Feed forward the input to the network to get the output matrix
        Mat output = net.forward();

        // Extract a 2D matrix for 4D output matrix with form of (number of detections x 7)
        Mat detections = new Mat(new Size(output.size(3), output.size(2)), CV_32F, output.ptr(0, 0));

        // Create indexer to access elements of the matrix
        FloatIndexer srcIndexer = detections.createIndexer();

        for (int i = 0; i < detections.size(0); i++) {
            float confidence = srcIndexer.get(i, 2);
            float f1 = srcIndexer.get(i, 3);
            float f2 = srcIndexer.get(i, 4);
            float f3 = srcIndexer.get(i, 5);
            float f4 = srcIndexer.get(i, 6);

            if (confidence > 0.55) {
                // Scale the coordinates to the original image size
                int x1 = Math.round(f1 * originalWidth);
                int y1 = Math.round(f2 * originalHeight);
                int x2 = Math.round(f3 * originalWidth);
                int y2 = Math.round(f4 * originalHeight);

                // Draw the rectangle around the detected face in green
                rectangle(image, new Rect(new Point(x1, y1), new Point(x2, y2)), new Scalar(0, 255, 0, 0), 2, LINE_8, 0);

                // Prepare the text with the confidence level
                String label = String.format("Do chinh xac: %d%%", Math.round(confidence * 100));

                // Calculate the position for the text
                int[] baseLine = new int[1];
                Size textSize = getTextSize(label, FONT_HERSHEY_TRIPLEX, 0.5, 1, baseLine);
                int textY = y1 - textSize.height();

                // Ensure the text is within image bounds
                if (textY < 0) {
                    textY = y1 + textSize.height();
                }

                // Draw the text above the rectangle in white
                putText(image, label, new Point(x1, textY), FONT_HERSHEY_TRIPLEX, 0.5, new Scalar(255, 255, 255, 0), 1, LINE_AA, false);
            }
        }
    }

    // Method to copy resource to a temporary file
    private static Path copyResourceToTempFile(String resourcePath) throws IOException {
        ClassLoader classLoader = FaceDetection.class.getClassLoader();
        try (InputStream resourceStream = classLoader.getResourceAsStream(resourcePath)) {
            if (resourceStream == null) {
                throw new IOException("Resource not found: " + resourcePath);
            }
            Path tempFile = Files.createTempFile("temp-", "-" + Path.of(resourcePath).getFileName().toString());
            Files.copy(resourceStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
            tempFile.toFile().deleteOnExit(); //temp file is deleted on exit
            return tempFile;
        }
    }
}
