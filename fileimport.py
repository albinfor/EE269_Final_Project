# Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Globals
class FeatureFile:
    def __init__(self, filename):
        self.test_percent = 0.20 # percent of dataset withheld for validation

        # Import Training Data
        # column names
        self.names = ['Subject ID',
                 'Jitter (local)', 'Jitter (local, abs)', 'Jitter (rap)',
                 'Jitter (ppq5)', 'Jitter (ddp)', 'Shimmer (local)',
                 'Shimmer (local, dB)', 'Shimmer (apq3)', 'Shimmer (apq5)',
                 'Shimmer (apq11)', 'Shimmer (dda)', 'AC', 'NTH', 'HTN',
                 'Median Pitch', 'Mean Pitch', 'Std Dev Pitch', 'Min Pitch',
                 'Max Pitch', 'Num Pulses', 'Num Periods', 'Mean Period',
                 'Std Dev Periods', 'Frac Unvoiced Frames', 'Num  Breaks',
                 'Degree of Breaks']
        # training column names
        train_names = self.names + ['UPDRS', 'class info']

        self.df = pd.read_csv(filename,
                         header=None,
                         names=train_names)
        self.df.head()
        # initialize patients arrary
        patients = {}
        for i in range(40):
            patients[i] = self.df.iloc[i * 26:i * 26 + 26].agg(['mean'])
        # remerge the averages
        avg_df = patients[0]
        for i in range(1, 40):
            avg_df = avg_df.append(patients[i])
        #self.df = avg_df

    def get_data_frame(self):
        # Get examples
        X = self.df.drop(['UPDRS', 'class info', 'Subject ID'], axis=1)
        # X = self.df.drop(['UPDRS', 'class info'], axis=1)
        # Get labels
        Y = self.df['class info']
        return [X, Y]

    def get_normalized_data_frame(self):
        # Get examples
        df = self.stats_norm_patients(self.df)
        df = pd.concat(df, axis=0)
        X = df.drop(['UPDRS', 'class info', 'Subject ID'], axis=1)
        # X = self.df.drop(['UPDRS', 'class info'], axis=1)
        # Get labels
        Y = df['class info']
        return [X, Y]

    def stats_norm_patients(self, df):
        return self.stats_patients(df, pat_func=self.normalize_patients)

    def stats_patients(self, df, pat_func):
        # get patients
        p = pat_func(df)
        # intialize stat based patients dictionary
        s = {}
        # for each patient
        for (k, v) in p.items():
            #print(k)
            s[k] = self.stats(v).drop(['Subject ID mean', 'UPDRS mean', 'class info mean',
                                  'Subject ID std', 'UPDRS std', 'class info std', ], axis=1)
            s[k]['Subject ID'] = v['Subject ID'].values[0]
            s[k]['UPDRS'] = v['UPDRS'].values[0]
            s[k]['class info'] = v['class info'].values[0]
        return s

    def normalize_patients(self, df):
        # remove labels and ID
        data = df.drop(['Subject ID', 'UPDRS', 'class info'], axis=1)
        # create Scaler
        scale = StandardScaler()
        # fit and transfrom the data
        normalized = pd.DataFrame(scale.fit_transform(data), columns=self.names[1:])
        # put labels and ID back in
        normalized['Subject ID'] = df['Subject ID']
        normalized['UPDRS'] = df['UPDRS']
        normalized['class info'] = df['class info']

        # break into patients and return
        return self.patients(normalized)

    def patients(self, df):
        p = {}
        for i in df['Subject ID'].unique():
            p[i - 1] = df.loc[df['Subject ID'] == i]
        return p

    def stats(self, df):
        # initialize features
        features = pd.DataFrame()
        # for each column in DataFrame
        for c in df.columns:
            # create a new feature of its mean
            features[c + ' mean'] = [df[c].mean(axis=0)]
            # create a new feature of its std
            features[c + ' std'] = [df[c].std(axis=0)]
        # return features
        return features