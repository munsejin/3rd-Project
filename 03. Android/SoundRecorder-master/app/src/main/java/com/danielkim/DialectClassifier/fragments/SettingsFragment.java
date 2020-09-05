package com.danielkim.DialectClassifier.fragments;

import android.os.Bundle;
import android.preference.Preference;
import android.preference.PreferenceFragment;
import android.support.annotation.Nullable;

import com.danielkim.DialectClassifier.BuildConfig;
import com.danielkim.DialectClassifier.R;
import com.danielkim.DialectClassifier.activities.SettingsActivity;

/**
 * Created by Daniel on 5/22/2017.
 * Modified by KCH on 13/08/2019.
 */

public class SettingsFragment extends PreferenceFragment {
    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(R.xml.preferences);

        Preference aboutPref = findPreference(getString(R.string.pref_about_key));
        aboutPref.setSummary(getString(R.string.pref_about_desc, BuildConfig.VERSION_NAME));
        aboutPref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener() {
            @Override
            public boolean onPreferenceClick(Preference preference) {
                LicensesFragment licensesFragment = new LicensesFragment();
                licensesFragment.show(((SettingsActivity)getActivity()).getSupportFragmentManager().beginTransaction(), "dialog_licenses");
                return true;
            }
        });

    }
}
