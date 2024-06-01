package teamti.fun;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;


public class Camera {
    public static void main(String[] args) {
        OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();

        final int RES_WIDTH = 1280;
        final int RES_HEIGHT = 720;

        VideoCapture capture = new VideoCapture();
        capture.set(CAP_PROP_FRAME_WIDTH, RES_WIDTH);
        capture.set(CAP_PROP_FRAME_HEIGHT, RES_HEIGHT);

        if (!capture.open(0)) {
            System.out.println("Can not open the cam !!!");
        }

        Mat colorimg = new Mat();

        CanvasFrame mainframe = new CanvasFrame("Face Detection", CanvasFrame.getDefaultGamma() / 2.2);
        mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        mainframe.setCanvasSize(RES_WIDTH, RES_HEIGHT);
        mainframe.setLocationRelativeTo(null);
        mainframe.setVisible(true);

        while (true) {
            while (capture.read(colorimg) && mainframe.isVisible()) {
                FaceDetection.detectAndDraw(colorimg);
                mainframe.showImage(converter.convert(colorimg));
            }
        }
    }
}
