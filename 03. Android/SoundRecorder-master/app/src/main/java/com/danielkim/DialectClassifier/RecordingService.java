package com.danielkim.DialectClassifier;

import android.app.Notification;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Environment;
import android.os.IBinder;
import android.os.Looper;
import android.support.v4.app.NotificationCompat;
import android.util.Log;

import android.widget.Toast;

import com.danielkim.DialectClassifier.activities.MainActivity;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Locale;
import java.util.Timer;
import java.util.TimerTask;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.Response;

/**
 * Created by Daniel on 12/28/2014.
 * Modified by KCH on 13/08/2019
 */
public class RecordingService extends Service {

    private static final String LOG_TAG = "RecordingService";

    private String mFileName = null;
    private String mFilePath = null;
    private AudioRecord mAudioRecord = null;

    private DBHelper mDatabase;

    private long mStartingTimeMillis = 0;
    private long mElapsedMillis = 0;
    private int mElapsedSeconds = 0;
    private OnTimerChangedListener onTimerChangedListener = null;
    private static final SimpleDateFormat mTimerFormat = new SimpleDateFormat("mm:ss", Locale.getDefault());
    private Timer mTimer = null;
    private TimerTask mIncrementTimerTask = null;

    private int mAudioSource = MediaRecorder.AudioSource.MIC;
    private int mSampleRate = 16000;
    private int mChannelCount = AudioFormat.CHANNEL_IN_MONO;
    private int mAudioFormat = AudioFormat.ENCODING_PCM_16BIT;
    private int mBufferSize = AudioRecord.getMinBufferSize(mSampleRate,
            mChannelCount, mAudioFormat);
    private Thread mRecordThread = null;
    private boolean isRecording = false;
    private String result = "1111";
    private HttpConnection httpConn = HttpConnection.getInstance();

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    public interface OnTimerChangedListener {
        void onTimerChanged(int seconds);
    }

    @Override
    public void onCreate() {
        super.onCreate();
        mDatabase = new DBHelper(getApplicationContext());
        Log.d("test", "create");
        mRecordThread = new Thread(new Runnable() {
            @Override
            public void run() {
                byte[] readData = new byte[mBufferSize];
                FileOutputStream fos = null;
                try {
                    fos = new FileOutputStream(mFilePath);

                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
                while (isRecording) {
                    int ret = mAudioRecord.read(readData, 0, mBufferSize);
                    Log.d("Write", "read bytes is " + ret);

                    try {
                        fos.write(readData, 0, mBufferSize);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                mAudioRecord.stop();
                mAudioRecord.release();
                mAudioRecord = null;
                try {
                    fos.close();
                    sendData();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

        startRecording();
        isRecording = true;
    }

    @Override
    public void onDestroy() {
        if (mAudioRecord != null) {
            stopRecording();
        }
        super.onDestroy();
    }

    public void startRecording() {
        setFileNameAndPath();
        mAudioRecord = new AudioRecord(mAudioSource, mSampleRate,
                mChannelCount, mAudioFormat, mBufferSize);
        mAudioRecord.startRecording();
        mStartingTimeMillis = System.currentTimeMillis();
        mRecordThread.start();
    }

    public void setFileNameAndPath() {
        int count = 0;
        File f;
        do {
            count++;
            mFileName = getString(R.string.default_file_name)
                    + "_" + (mDatabase.getCount() + count) + ".pcm";
            mFilePath = Environment.getExternalStorageDirectory().getAbsolutePath();
            mFilePath += "/Recordings/" + mFileName;

            f = new File(mFilePath);
        } while (f.exists() && !f.isDirectory());
    }

    public void stopRecording() {
        if (isRecording == true) isRecording = false;
        mElapsedMillis = (System.currentTimeMillis() - mStartingTimeMillis);
        //remove notification
        if (mIncrementTimerTask != null) {
            mIncrementTimerTask.cancel();
            mIncrementTimerTask = null;
        }

        try {
            mDatabase.addRecording(mFileName, mFilePath, mElapsedMillis);
        } catch (Exception e) {
            Log.e(LOG_TAG, "exception", e);
        }
    }

    private void startTimer() {
        mTimer = new Timer();
        mIncrementTimerTask = new TimerTask() {
            @Override
            public void run() {
                mElapsedSeconds++;
                if (onTimerChangedListener != null)
                    onTimerChangedListener.onTimerChanged(mElapsedSeconds);
                NotificationManager mgr = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
                mgr.notify(1, createNotification());
            }
        };
        mTimer.scheduleAtFixedRate(mIncrementTimerTask, 1000, 1000);
    }

    //TODO:
    private Notification createNotification() {
        NotificationCompat.Builder mBuilder =
                new NotificationCompat.Builder(getApplicationContext())
                        .setSmallIcon(R.drawable.ic_mic_white_36dp)
                        .setContentTitle(getString(R.string.notification_recording))
                        .setContentText(mTimerFormat.format(mElapsedSeconds * 1000))
                        .setOngoing(true);
        mBuilder.setContentIntent(PendingIntent.getActivities(getApplicationContext(), 0,
                new Intent[]{new Intent(getApplicationContext(), MainActivity.class)}, 0));
        return mBuilder.build();
    }

    private void sendData() {
        Log.d("test", "SendData");
        new Thread() {
            public void run() {
                httpConn.requestWebServer(mFilePath, callback);
            }
        }.start();
    }

    private final Callback callback = new Callback() {
        @Override
        public void onFailure(Call call, IOException e) {
            Log.d("test:", "콜백오류:" + e.getMessage());
        }

        @Override
        public void onResponse(Call call, Response response) throws IOException {
            result = response.body().string();
            clickMethod();
            Log.d("test:", "서버에서 응답한 Body:" + result);
        }
    };

    public void clickMethod() {
        Intent intent = new Intent("resultData");
        intent.putExtra("result", result);
        sendBroadcast(intent);
    }


}

