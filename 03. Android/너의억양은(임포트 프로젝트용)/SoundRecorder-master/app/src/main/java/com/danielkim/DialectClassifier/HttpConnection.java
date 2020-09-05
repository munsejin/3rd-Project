package com.danielkim.DialectClassifier;

import java.io.File;

import okhttp3.Callback;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;

public class HttpConnection {

    private OkHttpClient client;

    private static HttpConnection instance = new HttpConnection();
    public static HttpConnection getInstance() {
        return instance;
    }

    private HttpConnection(){ this.client = new OkHttpClient(); }

    public void requestWebServer(String mFilePath,Callback callback) {

        RequestBody requestBody = new MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("file","upload",
                        RequestBody.create(MultipartBody.FORM, new File(mFilePath)))
                .build();

        Request request = new Request.Builder()
                .url("http://192.168.0.24:5000/")
                .post(requestBody)
                .build();
        client.newCall(request).enqueue(callback);
    }

}