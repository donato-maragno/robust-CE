# # -*- coding: utf-8 -*-

import pandas as pd

### CLASSIFICATION

def banknote(wd): # Two classes
    """
    Attribute Information:
    1. variance of Wavelet Transformed image (continuous)
    2. skewness of Wavelet Transformed image (continuous)
    3. curtosis of Wavelet Transformed image (continuous)
    4. entropy of image (continuous)
    5. class (integer)
    """
    df = pd.read_csv(wd+'data_banknote_authentication.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['Outcome']
    return df
    

def ionosphere(wd): # Two classes
    """
    Attribute Information:

    -- All 34 are continuous
    -- The 35th attribute is either "good" or "bad" 
       according to the definition summarized above. 
       This is a binary classification task.
    """
    df = pd.read_csv(wd+'ionosphere.csv', header = None)
    df.iloc[:,-1] = (df.iloc[:,-1] == 'g')*1
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['Outcome']
    return df

def wdbc(wd): # Two classes
    """
    1) ID number
    2) Diagnosis (M = malignant, B = benign)
    3-32)
    Ten real-valued features are computed for each cell nucleus:
        a) radius (mean of distances from center to points on the perimeter)
        b) texture (standard deviation of gray-scale values)
        c) perimeter
        d) area
        e) smoothness (local variation in radius lengths)
        f) compactness (perimeter^2 / area - 1.0)
        g) concavity (severity of concave portions of the contour)
        h) concave points (number of concave portions of the contour)
        i) symmetry 
        j) fractal dimension ("coastline approximation" - 1)
    """
    df = pd.read_csv(wd+'wdbc.csv', header = None, index_col = 0)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns)-1)] 
    y = (df['y'] == 'M')*1
    df.drop('y', axis=1, inplace = True)
    df['Outcome'] = y
    return df


def diabetes(wd): # Two classes - Imbalanced
    """
    Attribute Information:
    1. Number of times pregnant
    2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    3. Diastolic blood pressure (mm Hg)
    4. Triceps skin fold thickness (mm)
    5. 2-Hour serum insulin (mu U/ml)
    6. Body mass index (weight in kg/(height in m)^2)
    7. Diabetes pedigree function
    8. Age (years)
    9. Class variable (0 or 1)
    """
    import pandas as pd
    df = pd.read_csv(wd+'diabetes.csv')
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['Outcome']
    return df


def phoneme(wd): # Two classes - Imbalanced
    """
    Attribute Information:
    Five different attributes were chosen to
    characterize each vowel: they are the amplitudes of the five first
    harmonics AHi, normalised by the total energy Ene (integrated on all the
    frequencies): AHi/Ene. Each harmonic is signed: positive when it
    corresponds to a local maximum of the spectrum and negative otherwise.
    6. Class (0 and 1)
    """
    import pandas as pd
    df = pd.read_csv(wd+'phoneme.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['Outcome']
    return df

