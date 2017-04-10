package com.cs.caine.superresolution;

import android.Manifest;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class SRMainActivity extends AppCompatActivity
{

    private static final  String TAG ="caine";

    private Bitmap bitmap, outBitmap;

    private static final String MODEL_FILE = "file:///android_asset/graph_valuble.pb";
    private static final String INPUT_NODE = "predict_input";
    private static final String OUTPUT_NODE = "tanh_out";

    private static final int HEIGHT = 256;       //输入图片的像素高
    private static final int WIDTH = 256;        //输入图片的像素宽
    private static final int CHANNEL = 3;        //输入图片的通道数：RGB
    private static final int OUT_HEIGHT = 1024;  //输出图片的高度
    private static final int OUT_WIDTH  = 1024;  //输出图片的宽度

    private  int[]  intValues = new int[HEIGHT * WIDTH];
    private  int[]  intOutValues = new int[OUT_HEIGHT * OUT_WIDTH];
    private float[] inputs = new float[HEIGHT * WIDTH * CHANNEL];                   //用于存储的模型输入数据
    private float[] outputs = new float[OUT_HEIGHT * OUT_WIDTH * CHANNEL];     //用于存储模型的输出数据
    private TensorFlowInferenceInterface inferenceInterface = new TensorFlowInferenceInterface();   //接口定义
    private  ImageView imageView = null;
    private  ImageView outView = null;
    Handler handler = null;
    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_srmain);

        Button buttonUpScale = (Button)findViewById(R.id.btnUpScale);
        imageView = (ImageView) findViewById(R.id.sample_view);
        outView = (ImageView) findViewById(R.id.outView) ;
        initPermission();



        handler = new Handler()
        {
            @Override
            public void handleMessage(Message msg)
            {
                super.handleMessage(msg);
                imageView.setImageBitmap(bitmap);
                outView.setImageBitmap(outBitmap);
            }
        };


        buttonUpScale.setOnClickListener(new Button.OnClickListener()
        {

            @Override
            public void onClick(View v)
            {
                int permission = ContextCompat.checkSelfPermission(SRMainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE);
                if (permission == PackageManager.PERMISSION_GRANTED)
                {


                    new Thread(new Runnable()
                    {
                        @Override
                        public void run()
                        {
                            bitmap = BitmapFactory.decodeFile("/sdcard/reference_small.png");

                            final long startTime = SystemClock.uptimeMillis();
                            UpSampleImage(bitmap);
                            final long lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                            System.out.println("Completeded!!!, cost " + String.valueOf(lastProcessingTimeMs) + "  ms");
                            handler.sendEmptyMessage(0);
                        }
                    }).start();

                }
                else
                {
                    Toast.makeText(getApplicationContext(),"Have no READ_EXTERNAL_STORAGE",Toast.LENGTH_LONG).show();
                }
            }
        });

    }


    public  void UpSampleImage(Bitmap bitmap)
    {
        //填充输入像素
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());

        Log.e(TAG, "Read input completed");

        for (int i = 0; i < intValues.length; ++i)
        {
            final int val = intValues[i];
//            inputs[i * 3 + 0] = ((val >> 16) & 0xFF)/255.0f ;
//            inputs[i * 3 + 1] = (((val >> 8) & 0xFF))/255.0f ;
//            inputs[i * 3 + 2] = (val & 0xFF) /255.0f ;
            inputs[i * 3 + 0] = ((val >> 16) & 0xFF) ;
            inputs[i * 3 + 1] = (((val >> 8) & 0xFF)) ;
            inputs[i * 3 + 2] = (val & 0xFF)  ;

        }

        Log.d(TAG,"Begin to load model");

        if (inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE) != 0)
        {
            Log.e("caine","TF initialization failed");
            throw new RuntimeException("TF initialization failed");
        }
        else
        {
            Log.e(TAG,"Model loaded");
        }

        inferenceInterface.fillNodeFloat(INPUT_NODE, new int[]{1, bitmap.getHeight(), bitmap.getWidth() ,3}, inputs);  //送入输入数据

        Log.e(TAG,"Data inputed,now inferencing....");

        inferenceInterface.runInference(new String[]{OUTPUT_NODE});     //进行模型的推理

        Log.e(TAG,"Inference completed, start output");

        inferenceInterface.readNodeFloat(OUTPUT_NODE, outputs);         //获取输出数据

        Log.e(TAG,"output completed");

        for (int i = 0; i < intOutValues.length; ++i)
        {
//            intOutValues[i] = 0xFF000000
//                            | (((int) (outputs[i * 3] * 255)) << 16)
//                            | (((int) (outputs[i * 3 + 1] * 255)) << 8)
//                            | ((int) (outputs[i * 3 + 2] * 255));
            intOutValues[i] = 0xFF000000
                    | (((int) (outputs[i * 3] )) << 16)
                    | (((int) (outputs[i * 3 + 1] )) << 8)
                    | ((int) (outputs[i * 3 + 2] ));
//            System.out.println( "r:" + String.valueOf(outputs[i * 3]) +";  g:" +String.valueOf((outputs[i * 3 + 1])
//                    +";  b:" + String.valueOf(outputs[i * 3 + 2])));
        }


        outBitmap = Bitmap.createBitmap(intOutValues,0,OUT_WIDTH,OUT_WIDTH,OUT_HEIGHT, Bitmap.Config.ARGB_8888);
      //  outBitmap = Bitmap.createBitmap(OUT_WIDTH,OUT_HEIGHT,Bitmap.Config.ARGB_8888);
       // outBitmap.setPixels(intOutValues, 0, OUT_WIDTH, 0, 0, OUT_WIDTH, OUT_HEIGHT);

        Log.e(TAG,"create output bitmap");
        saveBitmap(outBitmap);

       // imageView.setImageBitmap(outBitmap);
    }


    public void saveBitmap(Bitmap bm)
    {
        File f = new File("/sdcard/", "tensorflowtestout.png");
        if (f.exists())
        {
            f.delete();
        }
        try
        {
            FileOutputStream out = new FileOutputStream(f);
            bm.compress(Bitmap.CompressFormat.PNG, 90, out);
            out.flush();
            out.close();
            Log.i(TAG, "已经保存");
        }
        catch (FileNotFoundException e)
        {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        catch (IOException e)
        {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

    private void initPermission()
    {
        int permission = ContextCompat.checkSelfPermission(SRMainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE);

        if (permission != PackageManager.PERMISSION_GRANTED)
        {
            //需不需要解释的dialog
            if (shouldRequest()) return;
            //请求权限
            ActivityCompat.requestPermissions(SRMainActivity.this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE,
                            Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.MOUNT_UNMOUNT_FILESYSTEMS}, 1);
        }
    }

    private boolean shouldRequest()
    {
        if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.READ_EXTERNAL_STORAGE)) {
            //显示一个对话框，给用户解释
            explainDialog();
            return true;
        }
        return false;
    }

    private void explainDialog()
    {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage("app need permission,allow it？")
                .setPositiveButton("OK", new DialogInterface.OnClickListener()
                {
                    @Override
                    public void onClick(DialogInterface dialog, int which)
                    {
                        //请求权限
                        ActivityCompat.requestPermissions(SRMainActivity.this,
                                new String[]{Manifest.permission.READ_EXTERNAL_STORAGE,
                                        Manifest.permission.WRITE_EXTERNAL_STORAGE,
                                        Manifest.permission.MOUNT_UNMOUNT_FILESYSTEMS}, 1);
                    }
                }).setNegativeButton("Cancel", null)
                .create().show();
    }
}
