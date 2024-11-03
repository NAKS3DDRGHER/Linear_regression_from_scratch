import numpy as np
import pandas as pd

dev_type = [
    'Senior Executive (C-Suite, VP, etc.)', 'Developer, back-end',
    'Developer, front-end', 'Developer, full-stack',
    'Developer, QA or test', 'System administrator',
    'Database administrator',
    'Developer, desktop or enterprise applications',
    'Data or business analyst', 'Engineering manager',
    'Cloud infrastructure engineer',
    'Developer Experience',
    'Data scientist or machine learning specialist',
    'Academic researcher', 'Developer, mobile', 'Engineer, data',
    'Developer, embedded applications or devices',
    'Developer Advocate', 'Project manager', 'Security professional',
    'Hardware Engineer', 'Developer, game or graphics',
    'Product manager', 'DevOps specialist',
    'Research & Development role', 'Engineer, site reliability',
    'Designer', 'Blockchain', 'Scientist',
    'Marketing or sales professional', 'Educator', 'Student'
]
ed_lvl = [
    'Bachelor’s degree (B.A., B.S., B.Eng., etc.)',
    'Some college/university study without earning a degree',
    'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)',
    'Professional degree (JD, MD, Ph.D, Ed.D, etc.)',
    'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)',
    'Associate degree (A.A., A.S., etc.)',
    'Primary/elementary school'
]


language = [
    'Go', 'R', 'PowerShell', 'Apex', 'Scala',
    'Ruby', 'Groovy', 'Nim', 'Lisp', 'C',
    'Perl', 'Kotlin', 'Python', 'Clojure',
    'PHP', 'Bash/Shell (all shells)', 'Solidity',
    'APL', 'Prolog', 'Julia', 'Objective-C',
    'Assembly', 'Flow', 'Erlang', 'Raku',
    'Rust', 'GDScript', 'SAS', 'OCaml',
    'C#', 'Dart', 'TypeScript', 'JavaScript',
    'Elixir', 'Fortran', 'SQL', 'Swift',
    'Visual Basic (.Net)', 'Java', 'F#',
    'Lua', 'VBA', 'Ada', 'Cobol', 'Delphi',
    'Zig', 'C++', 'Crystal', 'MATLAB', 'HTML/CSS', 'Haskell'
]

def  z_score_normalization(x_train, y_train):
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    for i in range(x_train.shape[1]):
        if sigma[i] != 0:
            x_train[:,i] = (x_train[:,i] - mu[i]) / sigma[i]
    return x_train, y_train

def get_train_data():
    feature = ["YearsCode", "WorkExp", "YearsCodePro", "OrgSize", "Age", "PurchaseInfluence", "RemoteWork", "DevType", "EdLevel", "LanguageHaveWorkedWith"]
    df = pd.read_csv("data/survey_results_public.csv")
    history_df = pd.read_csv("data/history.csv")
    w, b = history_df.loc[len(history_df.index)-1, ["w", "b"]]
    w = np.array(w[1:-1].split(), dtype=float)
    filt = ((df["Country"] == "United States of America") & (df["CompTotal"] <= 300000))
    df = df.loc[filt, feature + ["CompTotal"]].replace(["Less than 1 year", 'More than 50 years'], [0, 50]).dropna().reset_index()
    df["OrgSize"] = parse_Orgsize(df)
    df["Age"] = parse_Age(df)
    df["PurchaseInfluence"] = parse_PurchaseInfluence(df)
    df["RemoteWork"] = parse_RemoteWork(df)
    df["DevType"] = parse_DevType(df)
    df["EdLevel"] = parse_EdLevel(df)
    df.reset_index(inplace=True)
    df = parse_LanguageHaveWorkedWith(df)
    feature.pop()
    pd.set_option('display.max_column', None)



    x_train = np.zeros([df.shape[0], len(feature) + len(language)])
    for idx, row in df.iterrows():
        x_train[idx] = np.array(row[feature + language], dtype="float128")

    y_train = np.array(df["CompTotal"], dtype="float128")

    return x_train, y_train, w, b

def upload_history(cur_w, cur_b):
    history_df = pd.read_csv("data/history.csv")
    history_df.loc[len(history_df.index)] = [cur_w, cur_b]
    history_df.to_csv("data/history.csv", index=False)

def parse_LanguageHaveWorkedWith(df):
    df['LanguageHaveWorkedWith'] = df['LanguageHaveWorkedWith'].str.split(";")
    for new in language:
        df[new] = 0

    for idx, row in df.iterrows():
        for lang in  row['LanguageHaveWorkedWith']:
            df.loc[idx, [lang]] = 1
    return df

def parse_Orgsize(df):
    df.drop(df[df['OrgSize'] == 'I don’t know'].index, inplace=True)
    df["OrgSize"] = df['OrgSize'].apply(convert_org_size)
    return df["OrgSize"]

def parse_EdLevel(df):
    df.drop(df[df['EdLevel'] == 'Something else'].index, inplace=True)
    df.replace(ed_lvl, range(7), inplace=True)
    return df["EdLevel"]

def parse_DevType(df):
    df.drop(df[df['DevType'] == 'Other (please specify):'].index, inplace=True)
    df.replace(dev_type, range(32), inplace=True)
    return df["DevType"]

def parse_Age(df):
    df.drop(df[df['Age'] == 'Prefer not to say'].index, inplace=True)
    df['Age'] = df['Age'].apply(convert_age)
    return df["Age"]

def parse_RemoteWork(df):
    df["RemoteWork"].replace(["Remote", "Hybrid (some remote, some in-person)", "In-person"], [0, 1, 2], inplace=True)
    return df["RemoteWork"]

def parse_PurchaseInfluence(df):
    df["PurchaseInfluence"].replace(["I have little or no influence", 'I have some influence', 'I have a great deal of influence'], [0, 1, 2], inplace=True)
    return df["PurchaseInfluence"]
    

def convert_org_size(size):
    if isinstance(size, str):
        size = size.replace(",", "")
        if 'or more' in size:
            return float(size.split()[0])
        elif 'Just me' in size:
            return 1.0
        else:
            range_values = size.split()[0:3:2]
            return (float(range_values[0]) + float(range_values[1])) / 2
    return size

def convert_age(age):
    if 'Under 18' in age:
        return 17
    elif '65 years or older' in age:
        return 65
    else:
        range_values = age.split('-')
        return (float(range_values[0]) + float(range_values[1].split()[0])) / 2
