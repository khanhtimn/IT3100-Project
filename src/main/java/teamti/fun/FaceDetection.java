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

        //resize the image to match the input size of the model
        resize(image, image, new Size(1280, 960));


        //create a 4-dimensional blob from image with NCHW (Number of images in the batch, Channel, Height, Width) dimensions order,
        //https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#gabd0e76da3c6ad15c08b01ef21ad55dd8
        Mat blob = blobFromImage(image, 1.0, new Size(640, 640), new Scalar(104.0, 177.0, 123.0, 0), false, false, CV_32F);

        //set the input to network model
        net.setInput(blob);

        //feed forward the input to the network to get the output matrix
        Mat output = net.forward();

        //extract a 2d matrix for 4d output matrix with form of (number of detections x 7)
        Mat ne = new Mat(new Size(output.size(3), output.size(2)), CV_32F, output.ptr(0, 0));

        //create indexer to access elements of the matrix
        FloatIndexer srcIndexer = ne.createIndexer();

        for (int i = 0; i < output.size(3); i++) {
            float confidence = srcIndexer.get(i, 2);
            float f1 = srcIndexer.get(i, 3);
            float f2 = srcIndexer.get(i, 4);
            float f3 = srcIndexer.get(i, 5);
            float f4 = srcIndexer.get(i, 6);
            if (confidence > .6) {
                float tx = f1 * 300;    //top-left point's x
                float ty = f2 * 300;    //top-left point's y
                float bx = f3 * 300;    //bottom-right point's x
                float by = f4 * 300;    //bottom-right point's y
                rectangle(image, new Rect(new Point((int) tx, (int) ty), new Point((int) bx, (int) by)), new Scalar(255, 0, 0, 0));
            }
        }
    }

    //TODO: Find a better solution for accessing resources in JAR files
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