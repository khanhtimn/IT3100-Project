package teamti.fun.facedetect;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;

import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class FaceDetectionApp extends Application {


    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(final Stage stage) throws Exception {
        stage.setTitle("Face Detection");
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("webcam.fxml"));
        Parent parent = fxmlLoader.load();
        final FaceDetectionController controller = fxmlLoader.getController();
        stage.setScene(new Scene(parent));
        stage.setMinWidth(300);
        stage.setMinHeight(300);
        stage.setOnCloseRequest(event -> {
            controller.shutdown();
            stage.close();
        });

        stage.show();
    }
}

