module teamti.fun.facedetect {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires java.desktop;
    requires java.logging;
    requires org.bytedeco.javacv;
    requires org.bytedeco.opencv;
    requires javafx.swing;
    requires org.bytedeco.opencv.windows.x86_64;
    requires org.bytedeco.openblas.windows.x86_64;

    opens teamti.fun.facedetect to javafx.fxml;
    exports teamti.fun.facedetect;
}