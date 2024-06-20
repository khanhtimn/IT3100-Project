package teamti.fun.facedetect;

import javafx.animation.AnimationTimer;
import javafx.application.Platform;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.*;
import javafx.scene.layout.StackPane;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import teamti.fun.facedetect.utils.CaffeNetHelper;

import java.net.URL;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.ResourceBundle;

import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGRA2BGR;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2BGRA;

public class FaceDetectionController implements Initializable {
    @FXML
    private ImageView videoView;

    @FXML
    private Canvas overlayCanvas;

    private final Mat javaCVMat = new Mat();
    private final Mat bgrMat = new Mat();
    private final WritablePixelFormat<ByteBuffer> formatByte = PixelFormat.getByteBgraPreInstance();
    private final OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
    private final SimpleBooleanProperty cameraActiveProperty = new SimpleBooleanProperty(true);
    private final OpenCVFrameGrabber frameGrabber = new OpenCVFrameGrabber(0);

    private List<double[]> prevCoordsList = new ArrayList<>();
    private boolean firstFrame = true;

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        // Bind Canvas and ImageView size to their parent
        videoView.fitWidthProperty().bind(((StackPane) videoView.getParent()).widthProperty());
        videoView.fitHeightProperty().bind(((StackPane) videoView.getParent()).heightProperty());

        overlayCanvas.widthProperty().bind(((StackPane) overlayCanvas.getParent()).widthProperty());
        overlayCanvas.heightProperty().bind(((StackPane) overlayCanvas.getParent()).heightProperty());

        startCamera();
    }

    private void startCamera() {
        try {
            frameGrabber.start();
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }

        AnimationTimer timer = new AnimationTimer() {
            @Override
            public void handle(long now) {
                if (cameraActiveProperty.get()) {
                    try {
                        Frame frame = frameGrabber.grab();
                        if (frame != null) {
                            updateView(frame);
                        }
                    } catch (FrameGrabber.Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        };
        timer.start();
    }

    private void updateView(Frame frame) {
        int width = frame.imageWidth;
        int height = frame.imageHeight;

        Mat mat = converter.convert(frame);
        opencv_imgproc.cvtColor(mat, javaCVMat, COLOR_BGR2BGRA);

        // Convert to 3 channels (BGR) before detection
        opencv_imgproc.cvtColor(javaCVMat, bgrMat, COLOR_BGRA2BGR);

        // Perform face detection and draw results
        Platform.runLater(() -> {
            GraphicsContext gc = overlayCanvas.getGraphicsContext2D();
            gc.clearRect(0, 0, overlayCanvas.getWidth(), overlayCanvas.getHeight());

            prevCoordsList = CaffeNetHelper.detectAndDrawWithSmoothTransition(bgrMat, gc, overlayCanvas.getWidth(), overlayCanvas.getHeight(), prevCoordsList, firstFrame);
            firstFrame = false;
        });

        // Convert back to BGRA for display
        opencv_imgproc.cvtColor(bgrMat, javaCVMat, COLOR_BGR2BGRA);

        // Create a WritableImage
        WritableImage writableImage = matToWritableImage(javaCVMat, width, height);

        Platform.runLater(() -> videoView.setImage(writableImage));
    }

    private WritableImage matToWritableImage(Mat mat, int width, int height) {
        byte[] buffer = new byte[width * height * 4];
        mat.data().get(buffer);

        WritableImage image = new WritableImage(width, height);
        PixelWriter pixelWriter = image.getPixelWriter();
        pixelWriter.setPixels(0, 0, width, height, formatByte, buffer, 0, width * 4);
        return image;
    }

    public void shutdown() {
        cameraActiveProperty.set(false);
    }
}
