import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

if __name__ == '__main__':
    train_path = sys.argv[1]
    train_df = pd.read_csv(train_path)

    features_to_drop = ['Survival_Months', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 
                        'Age', 'Blood_Pressure_Pulse', 'Performance_Status', 'Patient_ID']

    features_to_drop_existing = [feature for feature in features_to_drop if feature in train_df.columns]
    train_df = train_df.drop(columns=features_to_drop_existing, errors='ignore')

    X = train_df.drop('Stage', axis=1)
    y = train_df['Stage']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = [feature for feature in ['Gender', 'Smoking_History', 'Tumor_Location', 'Treatment', 'Ethnicity', 'Insurance_Type', 'Family_History'] if feature in X_train.columns]
    numerical_features = [feature for feature in ['Tumor_Size_mm', 'Hemoglobin_Level', 'White_Blood_Cell_Count', 'Platelet_Count', 'Albumin_Level', 
                          'Alkaline_Phosphatase_Level', 'Alanine_Aminotransferase_Level', 'Aspartate_Aminotransferase_Level', 
                          'Creatinine_Level', 'LDH_Level', 'Calcium_Level', 'Phosphorus_Level', 'Glucose_Level', 'Potassium_Level', 
                          'Sodium_Level', 'Smoking_Pack_Years'] if feature in X_train.columns]

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # # Random Forest Classifier
    clf_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=4
    )

    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf_rf)
    ])

    pipeline_rf.fit(X_train, y_train)
    y_pred_rf = pipeline_rf.predict(X_test)

    macro_f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    micro_f1_rf = f1_score(y_test, y_pred_rf, average='micro')
    print(f"Random Forest - Macro F1 score: {macro_f1_rf}")
    print(f"Random Forest - Micro F1 score: {micro_f1_rf}")
    # # Voting Classifier
    clf_svm = SVC(probability=True, C=0.1, kernel='rbf')
    clf_nb = GaussianNB()

    voting_clf = VotingClassifier(estimators=[
        ('svm', clf_svm),
        ('nb', clf_nb)
    ], voting='soft')

    pipeline_voting = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', voting_clf)
    ])

    pipeline_voting.fit(X_train, y_train)
    y_pred_voting = pipeline_voting.predict(X_test)

    macro_f1_voting = f1_score(y_test, y_pred_voting, average='macro')
    micro_f1_voting = f1_score(y_test, y_pred_voting, average='micro')
    print(f"Voting Classifier - Macro F1 score: {macro_f1_voting}")
    print(f"Voting Classifier - Micro F1 score: {micro_f1_voting}")
    # # Gradient Boosting Classifier
    clf_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

    pipeline_gb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf_gb)
    ])

    pipeline_gb.fit(X_train, y_train)
    y_pred_gb = pipeline_gb.predict(X_test)

    macro_f1_gb = f1_score(y_test, y_pred_gb, average='macro')
    micro_f1_gb = f1_score(y_test, y_pred_gb, average='micro')
    print(f"Gradient Boosting - Macro F1 score: {macro_f1_gb}")
    print(f"Gradient Boosting - Micro F1 score: {micro_f1_gb}")
    # Stacking Classifier
    estimators = [
        ('svm', clf_svm),
        ('gb', clf_gb),
        ('nb', clf_nb)
    ]
    clf_stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    pipeline_stack = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf_stack)
    ])

    pipeline_stack.fit(X_train, y_train)
    y_pred_stack = pipeline_stack.predict(X_test)

    macro_f1_stack = f1_score(y_test, y_pred_stack, average='macro')
    micro_f1_stack = f1_score(y_test, y_pred_stack, average='micro')
    print(f"Stacking Classifier - Macro F1 score: {macro_f1_stack}")
    print(f"Stacking Classifier - Micro F1 score: {micro_f1_stack}")

   
    
    
 