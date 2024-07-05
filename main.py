import sys
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

if __name__ == '__main__':
    data_path = sys.argv[1]
    data_frame = pd.read_csv(data_path)

    features_to_drop = ['Survival_Months', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 'Gender',
                        'Age', 'Blood_Pressure_Pulse', 'Performance_Status', 'Ethnicity', 'Insurance_Type', 'Patient_ID']

    features_to_drop_existing = [feature for feature in features_to_drop if feature in data_frame.columns]
    data_frame = data_frame.drop(columns=features_to_drop_existing, errors='ignore')

    X = data_frame.drop('Stage', axis=1)
    y = data_frame['Stage']

    # Podela inicijalnog skupa podataka na trening i test skup u razmeri 80:20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Izdvajanje kategorickih obelezja
    categorical_features = [feature for feature in ['Smoking_History', 'Tumor_Location', 'Treatment', 'Family_History'] if feature in X_train.columns]
    
    # Izdvajanje numerickih obelezja
    numerical_features = [feature for feature in ['Tumor_Size_mm', 'Hemoglobin_Level', 'White_Blood_Cell_Count', 'Platelet_Count', 'Albumin_Level', 
                          'Alkaline_Phosphatase_Level', 'Alanine_Aminotransferase_Level', 'Aspartate_Aminotransferase_Level', 
                          'Creatinine_Level', 'LDH_Level', 'Calcium_Level', 'Phosphorus_Level', 'Glucose_Level', 'Potassium_Level', 
                          'Sodium_Level', 'Smoking_Pack_Years'] if feature in X_train.columns]

    # Popunjavanje nedostajucih vrednosti srednjim vrednostima kolona i normalizacija za numericka obelezja
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Popunjavanje nedostajucih vrednosti najucestalijim vrednostima i one hot encoding za kategoricka obelezja
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

    # Boosting
    clf_gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.2)

    pipeline_gb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf_gb)
    ])

    pipeline_gb.fit(X_train, y_train)
    y_pred_gb = pipeline_gb.predict(X_test)

    macro_f1_gb = f1_score(y_test, y_pred_gb, average='macro')
    micro_f1_gb = f1_score(y_test, y_pred_gb, average='micro')
    precision_gb = precision_score(y_test, y_pred_gb, average='macro', zero_division=0)
    recall_gb = recall_score(y_test, y_pred_gb, average='macro', zero_division=0)

    print(f"Boosting - Macro F1 score: {macro_f1_gb}")
    print(f"Boosting - Micro F1 score: {micro_f1_gb}")
    print(f"Boosting - Precision: {precision_gb}")
    print(f"Boosting - Recall: {recall_gb}")
    print(f"Boosting - Classification Report:\n{classification_report(y_test, y_pred_gb, zero_division=0)}")

    # Stacking
    # Korisceni klasifikatori 
    clf_svm = SVC(probability=True, C=0.1, kernel='linear')
    clf_nb = GaussianNB()
    clf_knn = KNeighborsClassifier(weights='uniform', n_neighbors=7)

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

    pipeline_stack.fit(X_train, y_train)
    y_pred_stack = pipeline_stack.predict(X_test)

    macro_f1_stacking = f1_score(y_test, y_pred_stack, average='macro')
    micro_f1_stacking = f1_score(y_test, y_pred_stack, average='micro')
    precision_stack = precision_score(y_test, y_pred_stack, average='macro', zero_division=0)
    recall_stack = recall_score(y_test, y_pred_stack, average='macro', zero_division=0)

    print(f"Stacking - Macro F1 score: {macro_f1_stacking}")
    print(f"Stacking - Micro F1 score: {micro_f1_stacking}")
    print(f"Stacking - Precision: {precision_stack}")
    print(f"Stacking - Recall: {recall_stack}")
    print(f"Stacking - Classification Report:\n{classification_report(y_test, y_pred_stack, zero_division=0)}")

    # Optimizacija parametara
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
        n_iter=50,
        scoring='f1_macro',
        cv=3, 
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best macro F1 score: {random_search.best_score_}")

    best_model = random_search.best_estimator_
