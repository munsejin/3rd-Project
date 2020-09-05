package com.danielkim.DialectClassifier.fragments;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.support.v4.app.Fragment;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Chronometer;
import android.widget.TextView;
import android.widget.Toast;

import com.danielkim.DialectClassifier.R;
import com.danielkim.DialectClassifier.RecordingService;
import com.melnykov.fab.FloatingActionButton;

import java.io.File;

public class RecordFragment extends Fragment {
    private static final String ARG_POSITION = "position";
    private int position;
    private FloatingActionButton mRecordButton = null;
    private TextView mRecordingPrompt;
    private int mRecordPromptCount = 0;
    private boolean mStartRecording = true;
    private Chronometer mChronometer = null;
    long timeWhenPaused = 0;

    public static RecordFragment newInstance(int position) {
        RecordFragment f = new RecordFragment();
        Bundle b = new Bundle();
        b.putInt(ARG_POSITION, position);
        f.setArguments(b);
        return f;
    }

    public RecordFragment() {
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        position = getArguments().getInt(ARG_POSITION);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View recordView = inflater.inflate(R.layout.fragment_record, container, false);

        mChronometer = (Chronometer) recordView.findViewById(R.id.chronometer);
        //update recording prompt text
        mRecordingPrompt = (TextView) recordView.findViewById(R.id.recording_status_text);

        mRecordButton = (FloatingActionButton) recordView.findViewById(R.id.btnRecord);
        mRecordButton.setColorNormal(getResources().getColor(R.color.primary));
        mRecordButton.setColorPressed(getResources().getColor(R.color.primary_dark));
        mRecordButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onRecord(mStartRecording);
                mStartRecording = !mStartRecording;
            }
        });

        return recordView;
    }

    private void onRecord(boolean start) {

        Intent intent = new Intent(getActivity(), RecordingService.class);

        if (start) {
            mRecordButton.setImageResource(R.drawable.ic_media_stop);
            Toast.makeText(getActivity(), R.string.toast_recording_start, Toast.LENGTH_SHORT).show();
            File folder = new File(Environment.getExternalStorageDirectory() + "/Recordings");
            if (!folder.exists()) {
                folder.mkdir();
            }

            mChronometer.setBase(SystemClock.elapsedRealtime());
            mChronometer.start();
            mChronometer.setOnChronometerTickListener(new Chronometer.OnChronometerTickListener() {
                @Override
                public void onChronometerTick(Chronometer chronometer) {
                    if (mRecordPromptCount == 0) {
                        mRecordingPrompt.setText(getString(R.string.record_in_progress) + ".");
                    } else if (mRecordPromptCount == 1) {
                        mRecordingPrompt.setText(getString(R.string.record_in_progress) + "..");
                    } else if (mRecordPromptCount == 2) {
                        mRecordingPrompt.setText(getString(R.string.record_in_progress) + "...");
                        mRecordPromptCount = -1;
                    }

                    mRecordPromptCount++;
                }
            });

            getActivity().startService(intent);
            getActivity().getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

            mRecordingPrompt.setText(getString(R.string.record_in_progress) + ".");
            mRecordPromptCount++;
            registerReceiver();
        } else {

            mRecordButton.setImageResource(R.drawable.ic_mic_white_36dp);
            mChronometer.stop();
            mChronometer.setBase(SystemClock.elapsedRealtime());
            timeWhenPaused = 0;
            mRecordingPrompt.setText("분석중...");
            getActivity().stopService(intent);
            getActivity().getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        }

    }

    private BroadcastReceiver mReceiver = null;

    private void registerReceiver(){
        if(this.mReceiver != null) return;
        IntentFilter theFilter = new IntentFilter();
        theFilter.addAction("resultData");
        this.mReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                String receiveData = intent.getStringExtra("result");
                mRecordingPrompt.setText(receiveData);
                unregisterReceiver();
            }
        };
        getActivity().registerReceiver(this.mReceiver, theFilter);
    }
    private void unregisterReceiver() {
        if (this.mReceiver != null) {
            getActivity().unregisterReceiver(mReceiver);
            this.mReceiver = null;
        }
    }

}
