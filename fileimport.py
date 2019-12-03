# Libraries
import pandas as pd


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
        self.df = avg_df

    def get_data_frame(self):
        # Get examples
        X = self.df.drop(['UPDRS', 'class info', 'Subject ID'], axis=1)
        #X = self.df.drop(['UPDRS', 'class info'], axis=1)
        # Get labels
        Y = self.df['class info']
        return [X,Y]