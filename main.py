import sys
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

if __name__ == '__main__':
    data_path = sys.argv[1]
    data_frame = pd.read_csv(data_path)

    features_to_drop = ['Survival_Months', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 
                        'Age', 'Blood_Pressure_Pulse', 'Performance_Status', 'Patient_ID']

    features_to_drop_existing = [feature for feature in features_to_drop if feature in data_frame.columns]
    data_frame = data_frame.drop(columns=features_to_drop_existing, errors='ignore')

    X = data_frame.drop('Stage', axis=1)
    y = data_frame['Stage']

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

    clf_svm = SVC(probability=True, C=0.1, kernel='rbf')
    clf_nb = GaussianNB()
    clf_knn = KNeighborsClassifier()
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

    estimators = [
        ('svm', clf_svm),
        ('gb', clf_gb),
        ('knn', clf_knn)
    ]
    clf_stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    pipeline_stack = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf_stack)
    ])

    param_grid_svm = {
        'classifier__svm__C': [0.01, 0.1, 1, 10, 100],
        'classifier__svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'classifier__svm__gamma': ['scale', 'auto']
    }

    param_grid_knn = {
        'classifier__knn__n_neighbors': [3, 5, 7, 9, 11],
        'classifier__knn__weights': ['uniform', 'distance']
    }

    param_grid_gb = {
        'classifier__gb__n_estimators': [50, 100, 150],
        'classifier__gb__learning_rate': [0.01, 0.1, 0.2],
        'classifier__gb__max_depth': [3, 5, 7]
    }

    param_grid_stack = {
        'classifier__final_estimator__C': [0.01, 0.1, 1, 10, 100],
    }

    param_grid = {**param_grid_svm, **param_grid_knn, **param_grid_gb, **param_grid_stack}

    random_search = RandomizedSearchCV(
        estimator=pipeline_stack,
        param_distributions=param_grid,
        n_iter=50,  # Number of parameter settings sampled
        scoring='f1_macro',  # Evaluation metric
        cv=3,  # Number of folds for cross-validation
        verbose=2,  # Verbosity level
        n_jobs=-1,  # Use all available cores
        random_state=42
    )

    random_search.fit(X_train, y_train)

    # Best parameters and score
    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best macro F1 score: {random_search.best_score_}")

    best_model = random_search.best_estimator_

    y_pred_best = best_model.predict(X_test)
    macro_f1_best = f1_score(y_test, y_pred_best, average='macro')
    micro_f1_best = f1_score(y_test, y_pred_best, average='micro')

    print(f"Optimized Stacking Classifier - Macro F1 score: {macro_f1_best}")
    print(f"Optimized Stacking Classifier - Micro F1 score: {micro_f1_best}")
