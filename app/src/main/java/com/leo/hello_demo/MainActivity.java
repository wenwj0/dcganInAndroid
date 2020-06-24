package com.leo.hello_demo;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class MainActivity extends AppCompatActivity {

    private ImageView imageView;
    private Button button;
    private Button button2;
    private Net net;

    private void initView() {
        imageView = (ImageView) findViewById(R.id.imageView);
        button = (Button) findViewById(R.id.button);
        button2 = (Button) findViewById(R.id.button2);
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initView();
        //初始化
        if (OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCVLoader初始化成功", Toast.LENGTH_SHORT).show();
        }

        //生成图片
        button.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.O)
            @Override
            public void onClick(View v) {
                generateImage();
            }
        });
        //SAVE
        button2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                saveImage();
            }
        });

    }
    private void saveImage() {

    }
    @RequiresApi(api = Build.VERSION_CODES.O)
    private void generateImage() {
        //please put .prototxt and .caffemodel  in dcgan_demo\app\src\main\assets
        String proto = getPath("generator.prototxt", this);
        String weights = getPath("generator.caffemodel", this);
        net = Dnn.readNetFromCaffe(proto, weights);
        Mat input = new Mat();
        input = Mat.ones(64,100,CvType.CV_32FC1);
        Core.randn(input,0,1);
        net.setInput(input);

        List<Mat> list = new LinkedList<>();
        long startTime = System.nanoTime();
        Dnn.imagesFromBlob(net.forward(),list);
        long endTime = System.nanoTime();
        int ran = new Random().nextInt(list.size());
        Mat src = list.get(ran);
        int width = src.cols();
        int height = src.rows();
        double[] min = {Double.MAX_VALUE,Double.MAX_VALUE,Double.MAX_VALUE};
        double[] max = {Double.MIN_VALUE,Double.MIN_VALUE,Double.MIN_VALUE};

        for(int row=0; row<height; row++) {
            for (int col = 0; col < width; col++) {
                double[] pixel = src.get(row,col);
                for(int i=0;i<pixel.length;i++){
                    max[i] = pixel[i]>max[i]?pixel[i]:max[i];
                    min[i] = pixel[i]<min[i]?pixel[i]:min[i];
                }
            }
        }
        for(int row=0; row<height; row++) {
            for(int col=0; col<width; col++) {
                // 读取像素
                double[] pixel = src.get(row,col);
                for(int i=0;i<pixel.length;i++){
                    pixel[i] = (pixel[i] - min[i])/(max[i]-min[i]);
                }
                src.put(row,col,pixel);
            }
        }

        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        src.convertTo(src,CvType.CV_8U,255.0);
        Imgproc.cvtColor(src,src,Imgproc.COLOR_BGR2RGB);
        Utils.matToBitmap(src,bitmap);
        imageView.setImageBitmap(bitmap);

//        ImageView iv = (ImageView)this.findViewById(R.id.imageView2);
        String second = String.valueOf((endTime-startTime)/1000000);
        Toast.makeText(this, second, Toast.LENGTH_SHORT).show();
        src.release();
    }



    private static String getPath(String file, Context context){
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream= null;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            System.out.println(outFile.getAbsoluteFile());
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i("INFO", "Failed to upload a file");
        }
        return "";
    }
}
